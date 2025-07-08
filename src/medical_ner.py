import os
import json
import numpy as np
import torch
from datasets import load_dataset, Value, Features, ClassLabel, Sequence
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer, 
    DataCollatorForTokenClassification
)
import evaluate

# 设置环境变量
os.environ['MODELSCOPE_CACHE'] = "/root/autodl-tmp/.cache/hub"
os.environ['HF_HUB_CACHE'] = "/root/autodl-tmp/.cache/hub"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


class MedicalNERProcessor:
    """医疗命名实体识别数据处理类"""
    
    def __init__(self, data_dir='../data/medical'):
        """初始化处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.names = []
        self.id2label = {}
        self.label2id = {}
        
    def convert_txt_to_json(self, txt_path, json_path):
        """将文本格式的NER数据转换为JSON格式
        
        Args:
            txt_path: 输入文本文件路径
            json_path: 输出JSON文件路径
            
        Returns:
            更新后的标签列表
        """
        data = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for idx, text in enumerate(f.read().split('\n\n')):
                ner_tags = []
                sample = {}
                tokens = []
                for line in text.split('\n'):
                    if not line.strip():
                        continue
                    token_tag = line.split()
                    if len(token_tag) != 2:
                        continue
                    token, tag = token_tag
                    tokens.append(token)
                    if tag not in self.names:
                        self.names.append(tag)
                    ner_tags.append(self.names.index(tag))
                sample['id'] = idx
                sample['tokens'] = tokens
                sample['ner_tags'] = ner_tags
                data.append(sample)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return self.names
    
    def process_all_data(self):
        """处理所有数据集（训练集、验证集、测试集）"""
        # 处理训练集
        self.convert_txt_to_json(f'{self.data_dir}/train.txt', f'{self.data_dir}/train.json')
        print(f"训练集标签: {self.names}")
        
        # 处理验证集
        self.convert_txt_to_json(f'{self.data_dir}/dev.txt', f'{self.data_dir}/dev.json')
        print(f"添加验证集标签后: {self.names}")
        
        # 处理测试集
        self.convert_txt_to_json(f'{self.data_dir}/test.txt', f'{self.data_dir}/test.json')
        print(f"添加测试集标签后: {self.names}")
        
        # 创建标签映射
        self.id2label = {i: label for i, label in enumerate(self.names)}
        self.label2id = {label: i for i, label in enumerate(self.names)}
        
        print("id2label:", self.id2label)
        print("label2id:", self.label2id)
        
    def load_dataset(self):
        """加载处理后的数据集
        
        Returns:
            处理后的数据集
        """
        data_files = {
            'train': f'{self.data_dir}/train.json', 
            'dev': f'{self.data_dir}/dev.json', 
            'test': f'{self.data_dir}/test.json'
        }
        
        features = Features({
            'id': Value('int32'),
            'tokens': Sequence(Value('string')),
            'ner_tags': Sequence(ClassLabel(num_classes=len(self.names), names=self.names))
        })
        
        return load_dataset('json', data_files=data_files, features=features)


class MedicalNERModel:
    """医疗命名实体识别模型类"""
    
    def __init__(self, processor, checkpoint='Qwen/Qwen2.5-7B-Instruct', output_dir='./output'):
        """初始化模型
        
        Args:
            processor: 数据处理器实例
            checkpoint: 预训练模型检查点
            output_dir: 模型输出目录
        """
        self.processor = processor
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        self.seqeval = evaluate.load('seqeval')
        
    def load_model(self):
        """加载预训练模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.checkpoint, 
            num_labels=len(self.processor.names), 
            id2label=self.processor.id2label, 
            label2id=self.processor.label2id
        ).to(self.device)
        
        # 获取可训练参数数量
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print("可训练参数数量:", sum(p.numel() for p in trainable_params))
        
    def process_data(self, examples):
        """处理数据集的函数
        
        Args:
            examples: 数据样本
            
        Returns:
            处理后的样本
        """
        tokenized_examples = self.tokenizer(
            examples['tokens'], 
            truncation=True, 
            is_split_into_words=True, 
            max_length=512, 
            padding=True, 
            return_tensors='pt'
        )
        
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_examples.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            labels.append(label_ids)
        
        tokenized_examples['labels'] = labels
        return tokenized_examples
    
    def compute_metrics(self, p):
        """计算评估指标
        
        Args:
            p: 预测结果和标签
            
        Returns:
            评估指标字典
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.processor.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.processor.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = self.seqeval.compute(
            predictions=true_predictions, 
            references=true_labels, 
            mode="strict", 
            scheme="IOB2"
        )
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def train(self, raw_dataset, training_args=None):
        """训练模型
        
        Args:
            raw_dataset: 原始数据集
            training_args: 训练参数
        """
        # 处理数据集
        tokenized_dataset = raw_dataset.map(self.process_data, batched=True)
        print("处理后的数据集:", tokenized_dataset)
        
        # 设置默认训练参数
        if training_args is None:
            training_args = TrainingArguments(
                learning_rate=2e-5,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=8,
                num_train_epochs=3,
                weight_decay=0.01,
                output_dir=self.output_dir,
                remove_unused_columns=True,
                logging_steps=10,
                eval_strategy='epoch',
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                bf16=True,
                report_to="swanlab",
                run_name="medical_ner_1",
            )
        
        # 初始化训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_model(f"{self.output_dir}/model")
        print(f"模型已保存到 {self.output_dir}/model")


def main():
    """主函数"""
    # 初始化数据处理器
    processor = MedicalNERProcessor()
    
    # 处理数据
    processor.process_all_data()
    
    # 加载数据集
    raw_dataset = processor.load_dataset()
    
    # 初始化模型
    model = MedicalNERModel(processor)
    
    # 加载预训练模型
    model.load_model()
    
    # 训练模型
    model.train(raw_dataset)


if __name__ == "__main__":
    main()