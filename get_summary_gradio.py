import json
import os
import requests  # 导入 requests 库用于调用 DeepSeek API
from PIL import Image
from openai import OpenAI
import yaml
from surya.detection import batch_text_detection
from surya.input.load import load_from_folder, load_from_file
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor
from surya.ocr import run_ocr
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import gradio as gr


class GetSummary:
    def __init__(self, yaml_path='deepseek_api/config.yaml'):
        # 读取YAML文件
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        # 获取base_url和api_key
        base_url = config['base_url']
        api_key = config['api_key']
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
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
        return abstract_text1

    def get_layout_pred(self, image, name):
        layout_predictions = batch_layout_detection([image], self.layout_model, self.layout_processor)
        layout_predictions_bboxes = layout_predictions[0].bboxes
        for i in range(len(layout_predictions_bboxes) - 1):
            layout_label = layout_predictions_bboxes[i].label
            if layout_label == "SectionHeader" and layout_predictions_bboxes[i + 1].label in ["Text", "TextInlineMath"]:
                cropped_image = image.crop(layout_predictions_bboxes[i].bbox)
                ocr_res = self.ocr(cropped_image)
                if ocr_res[0].text.replace(" ", "").lower() in ["abstract", "abstrac"]:
                    if layout_predictions_bboxes[i + 1].height > 30:
                        cropped_text = image.crop(layout_predictions_bboxes[i + 1].bbox)
                        ocr_text = self.ocr(cropped_text)
                        abstract_text0 = "".join([line.text for line in ocr_text])
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
                cropped_image = image.crop(layout_predictions_bboxes[i].bbox)
                ocr_res = self.ocr(cropped_image)
                if "abstract" in ocr_res[0].text[:20].lower():
                    abstract_text0 = "".join([line.text for line in ocr_res])
                    abstract_text1 = self.get_second_summary(image, layout_predictions_bboxes[i + 1].label,
                                                             layout_predictions_bboxes[i + 1].height,
                                                             layout_predictions_bboxes[i + 1].bbox)
                    if abstract_text1:
                        all_abstract = abstract_text0 + "\n" + abstract_text1
                        return all_abstract
                    else:
                        return abstract_text0
        return None

    def ocr(self, image):
        predictions = run_ocr([image], [self.langs], self.det_model, self.det_processor, self.rec_model,
                              self.rec_processor, detection_batch_size=8,
                              recognition_batch_size=8)
        return predictions[0].text_lines

    def get_images(self, input_path):
        if os.path.isdir(input_path):
            images, names, _ = load_from_folder(input_path, 3)
        else:
            images, names, _ = load_from_file(input_path, 3)
        return images, names

    def out_summary(self, all_pdf_abstract, save_path, file_type="md"):
        if file_type == "json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_pdf_abstract, f, ensure_ascii=False, indent=4)
        if file_type == "md":
            with open(save_path, "w", encoding="utf-8") as f:
                for key, value in all_pdf_abstract.items():
                    f.write(f"{key}\n\n")
                    f.write(f"摘要：{value['original']}\n\n")
                    f.write(f"翻译：{value['translated']}\n\n")

    def run(self, input_path, out_path, file_type="md"):
        images, names = self.get_images(input_path)
        all_pdf_abstract = {}
        for i in range(0, len(images), 3):
            for j in range(i, i + 3):
                abstract = self.get_layout_pred(images[j], names[i])
                if abstract:
                    # 翻译摘要
                    translated_abstract = self.translate_text(abstract)
                    all_pdf_abstract[names[i]] = {
                        "original": abstract,
                        "translated": translated_abstract
                    }
                    break
            else:
                all_pdf_abstract[names[i]] = {
                    "original": "没有找到摘要",
                    "translated": "No abstract found"
                }
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        save_path = os.path.join(out_path, "output." + file_type)
        self.out_summary(all_pdf_abstract, save_path, file_type)
        return all_pdf_abstract

    # 翻译函数（使用 DeepSeek API）
    def translate_text(self, text):
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个中英文翻译专家，将用户输入的中文翻译成英文，或将用户输入的英文翻译成中文。对于非中文内容，它将提供中文翻译结果。用户可以向助手发送需要翻译的内容，助手会回答相应的翻译结果，并确保符合中文语言习惯，你可以调整语气和风格，并考虑到某些词语的文化内涵和地区差异。同时作为翻译家，需将原文翻译成具有信达雅标准的译文。\"信\" 即忠实于原文的内容与意图；\"达\" 意味着译文应通顺易懂，表达清晰；\"雅\" 则追求译文的文化审美和语言的优美。目标是创作出既忠于原作精神，又符合目标语言文化和读者审美的翻译。"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
            translated_text = completion.choices[0].message.content
            return translated_text
        except Exception as e:
            print(f"翻译失败：{e}")
            return "翻译失败"


# Gradio 界面
def process_pdf(files, save_path, file_type="md"):
    get_summary = GetSummary()
    all_results = {}
    for file in files:
        input_path = file.name  # Gradio 上传的文件是临时文件，通过 .name 获取路径
        result = get_summary.run(input_path, save_path, file_type)
        all_results.update(result)

    # 将所有结果写入一个文件
    output_file = os.path.join(save_path, "output." + file_type)
    if file_type == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
    elif file_type == "md":
        with open(output_file, "w", encoding="utf-8") as f:
            for key, value in all_results.items():
                f.write(f"{key}\n\n")
                f.write(f"摘要：{value['original']}\n\n")
                f.write(f"翻译：{value['translated']}\n\n")

    return all_results


# 格式化显示结果
def format_results(results):
    formatted_text = ""
    for key, value in results.items():
        formatted_text += f"{key}\n\n"
        formatted_text += f"摘要：{value['original']}\n\n"
        formatted_text += f"翻译：{value['translated']}\n\n"
    return formatted_text


# Gradio 应用
def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("## PDF 摘要提取工具")
        with gr.Row():
            file_input = gr.File(label="上传 PDF 文件", file_types=[".pdf"], file_count="multiple")  # 支持多文件上传
            file_type = gr.Radio(choices=["md", "json"], label="输出格式", value="md")
            save_path = gr.Textbox(label="保存路径", value=os.getcwd(), placeholder="请输入保存路径")  # 默认当前目录
            browse_button = gr.Button("选择保存路径")  # 添加浏览按钮
        submit_button = gr.Button("提取摘要")
        output = gr.Textbox(label="摘要结果", lines=10, interactive=False)  # 使用 Textbox 显示纯文本结果

        # 浏览按钮点击事件
        def browse_folder():
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口
            root.attributes("-topmost", True)  # 置顶窗口
            folder_path = filedialog.askdirectory()  # 打开文件夹选择对话框
            root.destroy()  # 关闭窗口
            return folder_path

        browse_button.click(
            lambda: browse_folder(),
            outputs=save_path
        )

        # 提交按钮点击事件
        submit_button.click(
            lambda files, save_path, file_type: format_results(process_pdf(files, save_path, file_type)),
            inputs=[file_input, save_path, file_type],
            outputs=output
        )

    demo.launch(share=True)  # 设置为 share=True 以生成公共链接


if __name__ == '__main__':
    gradio_app()
