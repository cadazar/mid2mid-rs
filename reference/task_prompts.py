#!/usr/bin/env python3
# coding: utf-8
"""
Task prompt library for CRAFT training.

Provides multilingual prompts for different task types,
with heavier sampling for Korean and English.
"""

import random
from typing import Literal, Optional

# language weights for sampling (Korean and English emphasized)
LANGUAGE_WEIGHTS = {
    'ko': 0.36,  # korean
    'en': 0.36,  # english
    'zh': 0.08,  # chinese
    'ja': 0.08,  # japanese
    'vi': 0.05,  # vietnamese
    'fr': 0.04,  # french
    'de': 0.03,  # german
}

# task prompts by language
TASK_PROMPTS = {
    'denoising': {
        'ko': [
            "다음 텍스트를 복원하시오:",
            "손상된 텍스트를 재구성하시오:",
            "빈칸을 채워 텍스트 데이터를 완성하시오:",
        ],
        'en': [
            "Reconstruct the following text:",
            "Fill in the missing parts:",
            "Restore the corrupted text:",
        ],
        'zh': [
            "重建以下文本：",
            "填补缺失部分：",
            "恢复损坏的文本：",
        ],
        'ja': [
            "次のテキストを復元してください：",
            "欠落部分を埋めてください：",
            "破損したテキストを修復してください：",
        ],
        'vi': [
            "Khoi phuc van ban sau day:",
            "Dien vao cac phan bi thieu:",
            "Phuc hoi van ban bi hu hong:",
        ],
        'fr': [
            "Reconstruisez le texte suivant :",
            "Remplissez les parties manquantes :",
            "Restaurez le texte corrompu :",
        ],
        'de': [
            "Rekonstruieren Sie den folgenden Text:",
            "Ergaenzen Sie die fehlenden Teile:",
            "Stellen Sie den beschaedigten Text wieder her:",
        ],
    },
    'ocr_correction': {
        'ko': [
            "OCR 오류를 수정하시오:",
            "잘못 인식된 텍스트를 교정하시오:",
            "스캔 오류를 바로잡으시오:",
        ],
        'en': [
            "Correct the OCR errors:",
            "Fix the misrecognized text:",
            "Repair scanning errors:",
        ],
        'zh': [
            "纠正OCR错误：",
            "修正误识别的文本：",
            "修复扫描错误：",
        ],
        'ja': [
            "OCRエラーを修正してください：",
            "誤認識されたテキストを訂正してください：",
            "スキャンエラーを修復してください：",
        ],
    },
    'temporal_classification': {
        'ko': [
            "이 텍스트가 작성된 연도를 추정하시오:",
            "다음 글의 시대를 판별하시오:",
            "문서의 작성 시기를 밝히시오:",
        ],
        'en': [
            "Estimate when this text was written:",
            "Determine the time period of this document:",
            "Identify the year of composition:",
        ],
        'zh': [
            "估计此文本的写作年代：",
            "确定此文档的时期：",
            "识别创作年份：",
        ],
        'ja': [
            "このテキストが書かれた年代を推定してください：",
            "この文書の時代を判定してください：",
            "作成時期を特定してください：",
        ],
        'vi': [
            "Uoc tinh thoi gian van ban nay duoc viet:",
            "Xac dinh thoi ky cua tai lieu nay:",
            "Nhan dien nam sang tac:",
        ],
        'fr': [
            "Estimez la date de redaction de ce texte :",
            "Determinez la periode de ce document :",
            "Identifiez l'annee de composition :",
        ],
        'de': [
            "Schaetzen Sie, wann dieser Text geschrieben wurde:",
            "Bestimmen Sie die Zeitperiode dieses Dokuments:",
            "Identifizieren Sie das Entstehungsjahr:",
        ],
    },
    'sts': {
        'ko': [
            "두 문장의 의미적 유사도를 평가하시오:",
            "문장 간 유사성을 0-5 척도로 측정하시오:",
            "다음 문장 쌍의 의미적 관련성을 판단하시오:",
        ],
        'en': [
            "Evaluate the semantic similarity of these sentences:",
            "Rate the similarity on a 0-5 scale:",
            "Assess the semantic relatedness of this sentence pair:",
        ],
        'zh': [
            "评估这些句子的语义相似度：",
            "用0-5量表评估相似度：",
            "判断这对句子的语义相关性：",
        ],
        'ja': [
            "これらの文の意味的類似度を評価してください：",
            "0-5のスケールで類似度を評価してください：",
            "この文ペアの意味的関連性を判断してください：",
        ],
    },
    'translation_ko_to_en': {
        'ko': [
            "다음을 영어로 번역하시오:",
            "다음 텍스트를 영어로 옮기시오:",
            "아래 한국어 문장을 영어로 번역하시오:",
        ],
        'en': [
            "Translate the following Korean to English:",
            "Convert this Korean text to English:",
            "Provide an English translation:",
        ],
        'zh': [
            "将以下韩语翻译成英语：",
            "将此韩语文本转换为英语：",
            "提供英文翻译：",
        ],
        'ja': [
            "次の韓国語を英語に翻訳してください：",
            "この韓国語テキストを英語に変換してください：",
            "英語訳を提供してください：",
        ],
    },
    'translation_en_to_ko': {
        'ko': [
            "다음을 한국어로 번역하시오:",
            "다음 영어 텍스트를 한국어로 옮기시오:",
            "아래 영어 문장을 한국어로 번역하시오:",
        ],
        'en': [
            "Translate the following English to Korean:",
            "Convert this English text to Korean:",
            "Provide a Korean translation:",
        ],
        'zh': [
            "将以下英语翻译成韩语：",
            "将此英语文本转换为韩语：",
            "提供韩文翻译：",
        ],
        'ja': [
            "次の英語を韓国語に翻訳してください：",
            "この英語テキストを韓国語に変換してください：",
            "韓国語訳を提供してください：",
        ],
    },
    'transcription_hanja_to_hangul': {
        'ko': [
            "한자를 한글로 전사하시오:",
            "다음 한문을 한글로 옮기시오:",
            "한자 표기를 한글 독음으로 변환하시오:",
        ],
        'en': [
            "Transcribe Hanja to Hangul:",
            "Convert the mixed script to pure Hangul:",
            "Remove Hanja and write in Hangul only:",
        ],
        'zh': [
            "将汉字转写为韩文：",
            "将混合文本转换为纯韩文：",
            "去除汉字，仅用韩文书写：",
        ],
        'ja': [
            "漢字をハングルに転写してください：",
            "混合文字を純粋なハングルに変換してください：",
            "漢字を削除してハングルのみで書いてください：",
        ],
    },
    'transcription_hangul_to_hanja': {
        'ko': [
            "한글을 한자로 전사하시오:",
            "다음 한글 텍스트에 적절한 한자를 추가하시오:",
            "한자 표기가 필요한 부분에 한자를 병기하시오:",
        ],
        'en': [
            "Transcribe Hangul to Hanja:",
            "Add appropriate Hanja to this Hangul text:",
            "Insert Hanja where needed in the text:",
        ],
        'zh': [
            "将韩文转写为汉字：",
            "在此韩文文本中添加适当的汉字：",
            "在需要的地方插入汉字：",
        ],
        'ja': [
            "ハングルを漢字に転写してください：",
            "このハングルテキストに適切な漢字を追加してください：",
            "必要な箇所に漢字を挿入してください：",
        ],
    },
    'continuation': {
        'ko': [
            "다음 텍스트를 이어서 작성하시오:",
            "문장을 완성하시오:",
            "이야기를 계속 써 내려가시오:",
        ],
        'en': [
            "Continue the following text:",
            "Complete the text:",
            "Continue the narrative:",
        ],
        'zh': [
            "继续以下文本：",
            "完成句子：",
            "继续叙述：",
        ],
        'ja': [
            "次のテキストを続けてください：",
            "文を完成させてください：",
            "物語を続けてください：",
        ],
        'vi': [
            "Tiep tuc van ban sau day:",
            "Hoan thanh doan van:",
            "Viet tiep cau chuyen:",
        ],
        'fr': [
            "Continuez le texte suivant :",
            "Completez le texte :",
            "Poursuivez le recit :",
        ],
        'de': [
            "Setzen Sie den folgenden Text fort:",
            "Vervollstaendigen Sie den Text:",
            "Fuehren Sie die Erzaehlung weiter:",
        ],
    },
    'morpheme_denoising': {
        'ko': [
            "띄어쓰기와 문장 구조를 복원하시오:",
            "공백 없는 텍스트에 올바른 띄어쓰기를 적용하시오:",
            "압축된 텍스트를 문장 단위로 재구성하시오:",
        ],
        'en': [
            "Restore spacing and sentence structure:",
            "Add proper word boundaries to compressed text:",
            "Reconstruct sentences from unspaced text:",
        ],
        'zh': [
            "恢复文本的分词和句子结构：",
            "为压缩文本添加正确的词边界：",
            "从无空格文本重建句子：",
        ],
        'ja': [
            "文章構造と単語区切りを復元してください：",
            "圧縮されたテキストに正しい分かち書きを適用してください：",
            "区切りのないテキストから文を再構成してください：",
        ],
    },
    'denoising_heavy': {
        'ko': [
            "심하게 손상된 텍스트를 복원하시오:",
            "대부분이 가려진 텍스트를 재구성하시오:",
            "절반 이상 손실된 텍스트를 복구하시오:",
        ],
        'en': [
            "Restore the heavily corrupted text:",
            "Reconstruct the extensively masked text:",
            "Recover text with substantial missing content:",
        ],
        'zh': [
            "恢复严重损坏的文本：",
            "重建大部分被遮蔽的文本：",
            "修复大量缺失内容的文本：",
        ],
        'ja': [
            "大幅に破損したテキストを復元してください：",
            "広範囲にマスクされたテキストを再構成してください：",
            "大部分が欠損したテキストを復旧してください：",
        ],
        'vi': [
            "Khoi phuc van ban bi hu hong nang:",
            "Tai tao van ban bi che khuat phan lon:",
            "Phuc hoi van ban bi mat noi dung dang ke:",
        ],
        'fr': [
            "Restaurez le texte fortement corrompu :",
            "Reconstruisez le texte masque en grande partie :",
            "Recuperez le texte avec un contenu manquant important :",
        ],
        'de': [
            "Stellen Sie den stark beschaedigten Text wieder her:",
            "Rekonstruieren Sie den umfangreich maskierten Text:",
            "Stellen Sie Text mit erheblich fehlendem Inhalt wieder her:",
        ],
    },
    'byte_reconstruction': {
        'ko': [
            "바이트 수준 텍스트를 복원하시오:",
            "바이트 단위로 손상된 텍스트를 재구성하시오:",
            "바이트 표현에서 원본 텍스트를 복구하시오:",
        ],
        'en': [
            "Reconstruct text from byte-level representation:",
            "Restore the byte-corrupted text:",
            "Recover original text from byte tokens:",
        ],
        'zh': [
            "从字节级表示重建文本：",
            "恢复字节损坏的文本：",
            "从字节令牌中恢复原始文本：",
        ],
        'ja': [
            "バイトレベル表現からテキストを再構成してください：",
            "バイト破損したテキストを復元してください：",
            "バイトトークンから元のテキストを復旧してください：",
        ],
        'vi': [
            "Tai tao van ban tu bieu dien muc byte:",
            "Phuc hoi van ban bi hu hong o muc byte:",
            "Khoi phuc van ban goc tu cac token byte:",
        ],
        'fr': [
            "Reconstruisez le texte a partir de sa representation en octets :",
            "Restaurez le texte corrompu au niveau des octets :",
            "Recuperez le texte original a partir des jetons d'octets :",
        ],
        'de': [
            "Rekonstruieren Sie den Text aus der Byte-Darstellung:",
            "Stellen Sie den auf Byte-Ebene beschaedigten Text wieder her:",
            "Stellen Sie den Originaltext aus Byte-Token wieder her:",
        ],
    },
    'classification': {
        'ko': [
            "다음 기사의 주제를 분류하시오:",
            "뉴스 기사의 분야를 판별하시오:",
            "텍스트의 카테고리를 결정하시오:",
        ],
        'en': [
            "Classify the topic of this article:",
            "Determine the category of this news:",
            "Identify the subject area:",
        ],
        'zh': [
            "对这篇文章的主题进行分类：",
            "判断这条新闻的类别：",
            "识别主题领域：",
        ],
        'ja': [
            "この記事のトピックを分類してください：",
            "このニュースのカテゴリを判定してください：",
            "主題分野を特定してください：",
        ],
    },
    'nli': {
        'ko': [
            "전제와 가설의 관계를 판단하시오:",
            "두 문장 간의 논리적 관계를 추론하시오:",
            "가설이 전제로부터 도출되는지 판단하시오:",
        ],
        'en': [
            "Determine the relationship between premise and hypothesis:",
            "Infer the logical relationship between the sentences:",
            "Judge whether the hypothesis follows from the premise:",
        ],
        'zh': [
            "判断前提与假设的关系：",
            "推断两句话之间的逻辑关系：",
            "判断假设是否由前提推出：",
        ],
        'ja': [
            "前提と仮説の関係を判断してください：",
            "二つの文の間の論理的関係を推論してください：",
            "仮説が前提から導かれるか判断してください：",
        ],
    },
    'temporal_continuation': {
        'ko': [
            "작성 시대를 고려하여 텍스트를 이어 쓰고, 작성 연도를 추정하시오:",
            "시대적 문체를 유지하며 계속 쓰고 연도를 밝히시오:",
            "문체를 분석하여 글을 이어가고, 작성 시기를 판단하시오:",
        ],
        'en': [
            "Continue the text considering its time period, then estimate the year:",
            "Write a period-appropriate continuation and identify the year of composition:",
            "Continue in the style of the era and determine when it was written:",
        ],
        'zh': [
            "考虑时代风格续写文本，并估计创作年份：",
            "以时代特征续写，判断作品年代：",
            "继续写作并分析文本的创作时期：",
        ],
        'ja': [
            "時代に合った文体で続きを書き、執筆年を推定してください：",
            "時代背景を考慮して続けて書き、作成年を特定してください：",
            "文体を分析して続きを書き、執筆時期を判断してください：",
        ],
        'vi': [
            "Tiep tuc van ban theo phong cach thoi ky va uoc tinh nam viet:",
            "Viet tiep phu hop voi thoi dai va xac dinh nam sang tac:",
            "Phan tich van phong de viet tiep va xac dinh thoi ky:",
        ],
        'fr': [
            "Continuez le texte en tenant compte de son epoque, puis estimez l'annee :",
            "Redigez une suite appropriee a la periode et identifiez l'annee de composition :",
            "Poursuivez dans le style de l'epoque et determinez la date de redaction :",
        ],
        'de': [
            "Setzen Sie den Text unter Beruecksichtigung der Epoche fort und schaetzen Sie das Jahr:",
            "Schreiben Sie eine zeitgemaesse Fortsetzung und bestimmen Sie das Entstehungsjahr:",
            "Analysieren Sie den Stil und schreiben Sie weiter, bestimmen Sie die Entstehungszeit:",
        ],
    },
    'multiple_choice': {
        'ko': [
            "다음 질문에 대한 정답을 선택하시오:",
            "보기 중 올바른 답을 고르시오:",
            "다음 문제의 답을 고르시오:",
        ],
        'en': [
            "Select the correct answer:",
            "Choose the right answer from the options:",
            "Answer the following question:",
        ],
        'zh': [
            "选择正确答案：",
            "从选项中选出正确答案：",
            "回答以下问题：",
        ],
        'ja': [
            "正しい答えを選んでください：",
            "選択肢から正解を選んでください：",
            "次の問題に答えてください：",
        ],
    },
}


def sample_task_prompt(
    task_type: Literal[
        'denoising', 'denoising_heavy', 'byte_reconstruction',
        'ocr_correction', 'temporal_classification', 'temporal_continuation',
        'sts', 'translation', 'transcription', 'continuation',
        'morpheme_denoising', 'translation_ko_to_en', 'translation_en_to_ko',
        'transcription_hanja_to_hangul', 'transcription_hangul_to_hanja',
        'multiple_choice',
    ],
    language: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple[str, str]:
    """
    Sample a task prompt with weighted language selection.

    Args:
        task_type: type of task (with direction for translation/transcription)
        language: force specific language (ko, en, zh, ja) or None for random
        seed: random seed for reproducibility

    Returns:
        tuple of (prompt string, resolved task direction)
    """
    if seed is not None:
        random.seed(seed)

    # handle generic translation/transcription by randomly choosing direction
    if task_type == 'translation':
        task_type = random.choice(['translation_ko_to_en', 'translation_en_to_ko'])
    elif task_type == 'transcription':
        task_type = random.choice(['transcription_hanja_to_hangul', 'transcription_hangul_to_hanja'])

    # select language
    if language is None:
        language = random.choices(
            list(LANGUAGE_WEIGHTS.keys()),
            weights=list(LANGUAGE_WEIGHTS.values()),
            k=1
        )[0]

    # select random prompt for this task and language
    # fall back to Korean if this task has no prompts for the sampled language
    task_langs = TASK_PROMPTS[task_type]
    if language not in task_langs:
        language = 'ko'
    prompts = task_langs[language]
    prompt = random.choice(prompts)

    return prompt, task_type


def add_task_prompt_to_example(
    example: dict,
    task_type: str,
    language: Optional[str] = None,
) -> tuple[dict, str]:
    """
    Add task prompt to example as metadata field.

    Args:
        example: data example dict
        task_type: type of task
        language: force specific language or None for random

    Returns:
        tuple of (example with 'metadata' field replaced with prompt, resolved task direction)
    """
    prompt, task_direction = sample_task_prompt(task_type, language=language)

    # replace existing metadata entirely
    example['metadata'] = prompt

    return example, task_direction


if __name__ == '__main__':
    # test prompt sampling
    print("Task Prompt Sampling Test")
    print("=" * 80)

    for task in TASK_PROMPTS.keys():
        print(f"\n{task.upper()}:")
        for lang in ['ko', 'en', 'zh', 'ja']:
            prompt = sample_task_prompt(task, language=lang)
            print(f"  [{lang}] {prompt}")

    print("\n" + "=" * 80)
    print("Weighted Random Sampling (100 samples):")
    lang_counts = {'ko': 0, 'en': 0, 'zh': 0, 'ja': 0}
    for _ in range(100):
        lang = random.choices(
            list(LANGUAGE_WEIGHTS.keys()),
            weights=list(LANGUAGE_WEIGHTS.values()),
            k=1
        )[0]
        lang_counts[lang] += 1

    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}% (expected: {int(LANGUAGE_WEIGHTS[lang]*100)}%)")