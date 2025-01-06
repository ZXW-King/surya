import json
import os

from PIL import Image
from openai import OpenAI

from surya.detection import batch_text_detection
from surya.input.load import load_from_folder, load_from_file
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor
from surya.ocr import run_ocr
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor


class GetSummary:
    def __init__(self, input_path):
        self.input_path = input_path
        self.langs = ["en"]
        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
        self.layout_model = load_layout_model()
        self.layout_processor = load_layout_processor()

    def get_second_summary(self, image, layout_label, layout_height, layout_bbox):
        abstract_text1 = ""
        if layout_label in ["Text", "TextInlineMath"] and layout_height > 40:
            cropped_text = image.crop(layout_bbox)
            ocr_text = self.ocr(cropped_text)
            abstract_text1 = "".join([line.text for line in ocr_text])
            # print(f"第二段摘要内容：\n{abstract_text1}")
        return abstract_text1

    def get_layout_pred(self, image, name):
        layout_predictions = batch_layout_detection([image], self.layout_model, self.layout_processor)
        layout_predictions_bboxes = layout_predictions[0].bboxes
        for i in range(len(layout_predictions_bboxes) - 1):
            layout_label = layout_predictions_bboxes[i].label
            if layout_label == "SectionHeader" and layout_predictions_bboxes[i + 1].label in ["Text", "TextInlineMath"]:
                # 明确有摘要标题的
                cropped_image = image.crop(layout_predictions_bboxes[i].bbox)
                ocr_res = self.ocr(cropped_image)
                # print(ocr_res[0].text.replace(" ", "").lower())
                if ocr_res[0].text.replace(" ", "").lower() in ["abstract", "abstrac"]:
                    # print(f"题目：{name}")
                    if layout_predictions_bboxes[i + 1].height > 30:
                        cropped_text = image.crop(layout_predictions_bboxes[i + 1].bbox)
                        ocr_text = self.ocr(cropped_text)
                        abstract_text0 = "".join([line.text for line in ocr_text])
                        # print(f"摘要内容：\n{abstract_text}")
                        try:
                            abstract_text1 = self.get_second_summary(image, layout_predictions_bboxes[i + 2].label,
                                                                     layout_predictions_bboxes[i + 2].height,
                                                                     layout_predictions_bboxes[i + 2].bbox)
                        except:
                            abstract_text1 = ""
                        if abstract_text1:
                            all_abstract = abstract_text0 + "\n" + abstract_text1
                            return all_abstract
                        else:
                            return abstract_text0
            elif layout_label in ["Text", "TextInlineMath"] and layout_predictions_bboxes[i].height > 50:
                # 段落中包含摘要的
                cropped_image = image.crop(layout_predictions_bboxes[i].bbox)
                ocr_res = self.ocr(cropped_image)
                if "abstract" in ocr_res[0].text[:20].lower():
                    # print(f"题目：{name}")
                    abstract_text0 = "".join([line.text for line in ocr_res])
                    # print(f"摘要内容：\n{abstract_text0}")
                    abstract_text1 = self.get_second_summary(image, layout_predictions_bboxes[i + 1].label,
                                                             layout_predictions_bboxes[i + 1].height,
                                                             layout_predictions_bboxes[i + 1].bbox)
                    # print(f"第二段摘要内容：\n{abstract_text1}")
                    if abstract_text1:
                        all_abstract = abstract_text0 + "\n" + abstract_text1
                        return all_abstract
                    else:
                        return abstract_text0
        return None

    def ocr(self, image):
        # Replace with your languages - optional but recommended
        predictions = run_ocr([image], [self.langs], self.det_model, self.det_processor, self.rec_model,
                              self.rec_processor, detection_batch_size=8,
                              recognition_batch_size=8)
        return predictions[0].text_lines
        # print(f"段落：{line.text}")

    def get_images(self):
        if os.path.isdir(self.input_path):
            images, names, _ = load_from_folder(self.input_path, 3)
            folder_name = os.path.basename(self.input_path)
        else:
            images, names, _ = load_from_file(self.input_path, 3)
            folder_name = os.path.basename(self.input_path).split(".")[0]
        return images, names

    def out_summary(self, all_pdf_abstract, save_path, file_type="md"):
        if file_type == "json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_pdf_abstract, f, ensure_ascii=False, indent=4)
        if file_type == "md":
            # 写入markdown
            with open(save_path, "w", encoding="utf-8") as f:
                for key, value in all_pdf_abstract.items():
                    f.write(f"## {key}\n\n")
                    f.write(f"**摘要：**{value}\n\n")

    def run(self, out_path, file_type="md"):
        images, names = self.get_images()
        all_pdf_abstract = {}
        for i in range(0, len(images), 3):
            for j in range(i, i + 3):
                abstract = self.get_layout_pred(images[j], names[i])
                if abstract:
                    all_pdf_abstract[names[i]] = abstract
                    break
            else:
                print(f"{names[i]}--没有找到摘要")
                all_pdf_abstract[names[i]] = "没有找到摘要"
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        save_path = os.path.join(out_path, "output." + file_type)
        self.out_summary(all_pdf_abstract, save_path, file_type)


if __name__ == '__main__':
    input_path = "./test_data/pdf"
    out_path = "./test_data/summary"
    get_summary = GetSummary(input_path)
    get_summary.run(out_path)


