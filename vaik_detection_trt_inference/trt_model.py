from typing import List, Dict, Tuple

from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex


class TrtModel:
    def __init__(self, input_saved_model_path: str = None, classes: Tuple = None):
        self.classes = classes
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(input_saved_model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def inference(self, input_image_list: List[np.ndarray], score_th: float = 0.2, nms_th: float = 0.5) -> Tuple[
        List[Dict], Dict
    ]:
        resized_image_array, resized_scales_list = self.__preprocess_image_list(input_image_list,
                                                                                self.inputs[0]['shape'][1:3])
        raw_pred = self.__inference(resized_image_array)
        filter_pred_list = self.__raw_pred_parse(raw_pred)
        filter_pred_list = self.__filter_score(filter_pred_list, score_th)
        if nms_th is not None:
            filter_pred_list = self.__filter_nms(filter_pred_list, nms_th)
        objects_dict_list = self.__output_parse(filter_pred_list, resized_scales_list,
                                                [input_image.shape[0:2] for input_image in input_image_list])
        return objects_dict_list, raw_pred

    def __inference(self, resize_input_tensor: np.ndarray) -> List[np.ndarray]:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resize_input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resize_input_tensor.dtype}')

        model_input_dtype = self.inputs[0]['dtype']
        output_spec_list = self.__output_spec()
        output_tensor_list = [np.zeros((resize_input_tensor.shape[0], *output_spec[0][1:]), output_spec[1]) for
                              output_spec in output_spec_list]
        for index in range(0, resize_input_tensor.shape[0], self.inputs[0]['shape'][0]):
            batch = resize_input_tensor[index:index + self.inputs[0]['shape'][0], :, :, :]
            batch_pad = np.zeros(self.inputs[0]['shape'], model_input_dtype)
            batch_pad[:batch.shape[0], :, :, :] = batch.astype(model_input_dtype)
            output_tensor = self.__inference_tensor(batch_pad)
            for tensor_index, output_tensor_elem in enumerate(output_tensor):
                output_tensor_list[tensor_index][index:index + batch.shape[0]] = output_tensor_elem[:batch.shape[0]]
        return output_tensor_list

    def __inference_tensor(self, input_array: np.ndarray):
        # Copy I/O and Execute
        cuda.memcpy_htod(self.inputs[0]['allocation'], input_array)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs]

    def __output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def __preprocess_image_list(self, input_image_list: List[np.ndarray], resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, List[Tuple[float, float]]]:
        resized_image_list = []
        resized_scales_list = []
        for input_image in input_image_list:
            resized_image, resized_scales = self.__preprocess_image(input_image, resize_input_shape)
            resized_image_list.append(resized_image)
            resized_scales_list.append(resized_scales)
        return np.stack(resized_image_list), resized_scales_list

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]),
                                dtype=input_image.dtype)
        resized_scale = min(resize_input_shape[1] / input_image.shape[1],
                            resize_input_shape[0] / input_image.shape[0])
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image, (resize_input_shape[1] / resized_scale, resize_input_shape[0] / resized_scale)

    def __raw_pred_parse(self, raw_pred: List[np.ndarray]):
        filter_pred_list = []
        for index in range(raw_pred[0].shape[0]):
            filter_pred = {}
            filter_pred['detection_boxes'] = raw_pred[1][index]
            filter_pred['detection_classes'] = raw_pred[3][index]
            filter_pred['detection_scores'] = raw_pred[2][index]
            filter_pred['num_detections'] = raw_pred[0][index]
            filter_pred_list.append(filter_pred)
        return filter_pred_list

    def __output_parse(self, pred_list: List[Dict], resized_scales_list: List[Tuple[int, int]],
                       image_shape_list: List[Tuple[int, int]]) -> List[List[Dict]]:
        objects_dict_list_list = []
        for index, pred in enumerate(pred_list):
            objects_dict_list = []
            for pred_index in range(pred['num_detections']):
                classes_index = int(pred['detection_classes'][pred_index])
                name = str(classes_index) if self.classes is None else self.classes[classes_index]
                ymin = max(0, int((pred['detection_boxes'][pred_index][0] * resized_scales_list[index][0])))
                xmin = max(0, int((pred['detection_boxes'][pred_index][1] * resized_scales_list[index][1])))
                ymax = min(image_shape_list[index][0] - 1,
                           int((pred['detection_boxes'][pred_index][2] * resized_scales_list[index][0])))
                xmax = min(image_shape_list[index][1] - 1,
                           int((pred['detection_boxes'][pred_index][3] * resized_scales_list[index][1])))
                object_extend_dict = {'score': pred['detection_scores'][pred_index]}
                objects_dict = pascal_voc_rw_ex.get_objects_dict_template(name, xmin, ymin, xmax, ymax,
                                                                          object_extend_dict=object_extend_dict)
                objects_dict_list.append(objects_dict)
            objects_dict_list_list.append(objects_dict_list)
        return objects_dict_list_list

    @classmethod
    def __filter_score(self, pred_list, score_th):
        filter_pred_list = []
        for pred in pred_list:
            mask = pred['detection_scores'] > score_th
            filter_pred = {}
            filter_pred['detection_boxes'] = pred['detection_boxes'][mask]
            filter_pred['detection_classes'] = pred['detection_classes'][mask]
            filter_pred['detection_scores'] = pred['detection_scores'][mask]
            filter_pred['num_detections'] = int(filter_pred['detection_classes'].shape[0])
            filter_pred_list.append(filter_pred)
        return filter_pred_list

    # Ref. https://python-ai-learn.com/2021/02/14/nmsfast/
    @classmethod
    def __calc_iou(cls, source_array, dist_array, source_area, dist_area):
        x_min = np.maximum(source_array[0], dist_array[:, 0])
        y_min = np.maximum(source_array[1], dist_array[:, 1])
        x_max = np.minimum(source_array[2], dist_array[:, 2])
        y_max = np.minimum(source_array[3], dist_array[:, 3])
        w = np.maximum(0, x_max - x_min + 0.0000001)
        h = np.maximum(0, y_max - y_min + 0.0000001)
        intersect_area = w * h
        iou = intersect_area / (source_area + dist_area - intersect_area)
        return iou

    # Ref. https://python-ai-learn.com/2021/02/14/nmsfast/
    @classmethod
    def __filter_nms(cls, pred_list, nms_th):
        filter_pred_list = []
        for pred in pred_list:
            bboxes = pred['detection_boxes']
            areas = ((bboxes[:, 2] - bboxes[:, 0] + 0.0000001) * (bboxes[:, 3] - bboxes[:, 1] + 0.0000001))
            sort_index = np.argsort(pred['detection_scores'])
            i = -1
            while (len(sort_index) >= 2 - i):
                max_scr_ind = sort_index[i]
                ind_list = sort_index[:i]
                iou = cls.__calc_iou(bboxes[max_scr_ind], bboxes[ind_list], areas[max_scr_ind], areas[ind_list])
                del_index = np.where(iou >= nms_th)
                sort_index = np.delete(sort_index, del_index)
                i -= 1
            filter_pred = {}
            filter_pred['detection_boxes'] = pred['detection_boxes'][sort_index][::-1]
            filter_pred['detection_classes'] = pred['detection_classes'][sort_index][::-1]
            filter_pred['detection_scores'] = pred['detection_scores'][sort_index][::-1]
            filter_pred['num_detections'] = int(filter_pred['detection_classes'].shape[0])
            filter_pred_list.append(filter_pred)
        return filter_pred_list