"""Gradio-интерфейс для пайплайна размытия волос.

Запуск:
    python app.py            # http://127.0.0.1:7860
    python app.py --share    # публичная Gradio-ссылка
"""

from __future__ import annotations

import argparse
import os
import threading
from typing import Optional

# SAM 3 на macOS падает на одном matmul внутри MPS (mixed dtype). Включаем
# fallback на CPU для ops, которые ещё не реализованы в MPS — на случай,
# если кто-то всё-таки переключит device на mps в UI.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
from PIL import Image

from pipeline import (
    HairBlurPipeline,
    PIPELINE_MODES,
    SAPIENS_CHECKPOINTS,
    pick_device,
)


# Кэшируем пайплайн под каждый набор параметров — чтобы не перегружать
# тяжёлые веса при каждом нажатии «Применить». Перезагрузка только при
# смене размера модели / сегментатора / матирования / устройства.
_PIPE_LOCK = threading.Lock()
_PIPE: Optional[HairBlurPipeline] = None
_PIPE_KEY: Optional[tuple] = None


def get_pipeline(sapiens_size: str, segmenter: str, matter: str,
                 device: str, feather_radius: float,
                 sam3_version: str, mode: str) -> HairBlurPipeline:
    global _PIPE, _PIPE_KEY
    key = (sapiens_size, segmenter, matter, device,
           round(feather_radius, 2), sam3_version, mode)
    with _PIPE_LOCK:
        if _PIPE is None or _PIPE_KEY != key:
            _PIPE = HairBlurPipeline(
                sapiens_size=sapiens_size,
                prefer_segmenter=segmenter,
                prefer_matter=matter,
                device=device,
                feather_radius=feather_radius,
                sam3_version=sam3_version,
                mode=mode,
            )
            _PIPE_KEY = key
        return _PIPE


def process(image, blur_radius, mode, segmenter, sapiens_size, sam3_version,
            matter, feather_radius, device_choice, do_clean, check_woman,
            progress=gr.Progress(track_tqdm=False)):
    if image is None:
        raise gr.Error("Сначала загрузите изображение.")

    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert("RGB")

    device = None if device_choice == "auto" else device_choice

    progress(0.05, desc="Загружаю модели…")
    pipe = get_pipeline(sapiens_size, segmenter, matter, device or "",
                        feather_radius, sam3_version, mode)

    if check_woman:
        progress(0.2, desc="Проверяю, что на фото женщина…")
    progress(0.4, desc=f"Сегментация ({pipe.segmenter.name})…")
    progress(0.7, desc=f"Матирование ({pipe.matter.name})…")
    result = pipe(image, blur_radius=blur_radius, do_clean=do_clean,
                  check_woman=check_woman)
    progress(1.0, desc="Готово")

    info_lines = [
        f"**Режим:** `{result.mode}`",
        f"**Сегментатор:** `{result.segmenter}`",
        f"**Матирование:** `{result.matter}`",
        f"**Покрытие маски:** {result.coverage:.2%}",
        f"**Размер изображения:** {image.width}×{image.height}",
    ]
    if result.woman_check is not None:
        ok = "✓" if result.woman_check >= 0.01 else "⚠"
        info_lines.insert(
            1, f"**Проверка «женщина»:** {ok} покрытие {result.woman_check:.2%}"
        )
    info = "  \n".join(info_lines)
    if result.coverage < 0.001:
        info += "\n\n⚠ Маска практически пустая — модель не нашла целевую " \
                "область. Смените сегментатор или фото."
    if result.woman_check is not None and result.woman_check < 0.01:
        info += "\n\n⚠ На фото не найден женский силуэт (промпт «woman»). " \
                "Размытие всё равно применено."
    return result.output, result.mask, result.alpha, info


# ─────────── описания всех настроек, выводятся под заголовком ───────────

PARAM_HELP = """
### Что настраивает каждый параметр

- **Радиус блюра** — сила размытия в пикселях. Малые значения (5–15) только
  смягчают волосы; средние (20–40) дают заметный «motion-blur»; большие
  (60–100) превращают волосы в почти однородное цветное пятно.
- **Сегментатор** — какая нейросеть выделяет волосы. Цепочка fallback —
  если выбранный не загрузится, автоматически переключится на следующий:
  - `sam3` — Meta SAM 3 / 3.1, open-vocabulary, текстовый промпт «hair».
    Загружается через официальный пакет `facebookresearch/sam3`. Веса
    гейтед: нужен HF-токен и одобренный доступ на `facebook/sam3.1`
    (или `facebook/sam3` для старой версии).
  - `sapiens` — Meta Sapiens (ECCV 2024), обученная только на людях.
    Очень точна на сложных причёсках; не требует доступа.
  - `segformer` — `jonathandinu/face-parsing` (B5). Лёгкий, быстрый,
    маска грубее.
- **Версия SAM 3** — `sam3.1` (март 2026, Object Multiplex, лучше для
  видео) или `sam3` (ноябрь 2025, оригинал). На статичном фото разница
  околонулевая.
- **Размер Sapiens** — нужен только если SAM 3 недоступен и пайплайн
  упал на Sapiens. `2b` самая точная (≈8 ГБ весов), `1b` — компромисс
  (≈4 ГБ), `0.6b` / `0.3b` — для слабого железа.
- **Матирование** — что делает с границей маски для плавного перехода:
  - `matanyone` — нейросетевой alpha-matting (CVPR 2025). Обрабатывает
    отдельные пряди и пушок, переход естественный.
  - `feather` — простое гауссово размытие самой маски; работает всегда,
    но края «мыльные».
- **Feather радиус** — сила размытия для fallback-матирования. На MatAnyone
  не влияет.
- **Устройство** — `auto` выберет CUDA → MPS → CPU. Принудительно `cpu`
  ставить только при ошибках на GPU.
- **Чистить маску** — пост-обработка через OpenCV: удаляет точечный шум
  и закрывает дыры. Обычно стоит держать включённым.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Harami Hair Blur", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🪞 Harami Hair Blur\n"
            "Сегментирует волосы человека на фотографии и аккуратно размывает их.  \n"
            "Стек: **SAM 3.1** (Meta, март 2026) с текстовым промптом «hair» "
            "через официальный пакет `facebookresearch/sam3` + "
            "**MatAnyone** (CVPR 2025) для alpha-matting на отдельных прядях.  \n"
            "Каскадный fallback: SAM 3 → Sapiens → SegFormer, "
            "и MatAnyone → feather, чтобы интерфейс всегда отвечал."
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    label="Входное изображение (перетащите сюда или кликните)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=420,
                )

                with gr.Accordion("⚙ Настройки", open=True):
                    mode = gr.Radio(
                        choices=list(PIPELINE_MODES),
                        value="hair",
                        label="Режим",
                        info="hair — размыть только волосы. "
                             "modesty — размыть волосы + любые открытые "
                             "участки кожи (плечи, руки, шея, ноги, торс). "
                             "Лицо остаётся чётким. Не работает на segformer.",
                    )
                    check_woman = gr.Checkbox(
                        value=False,
                        label="Сначала проверить, что на фото женщина",
                        info="Дополнительный прогон SAM 3 с промптом «woman» "
                             "(+30 сек на CPU). Если силуэт не найден — "
                             "выводит предупреждение, размытие всё равно "
                             "применяется. Работает только когда сегментатор = sam3.",
                    )
                    blur_radius = gr.Slider(
                        1, 100, value=25, step=1,
                        label="Радиус блюра (пиксели)",
                        info="Сила размытия. 5–15 — лёгкое смягчение, "
                             "20–40 — заметный блюр, 60+ — однородное пятно.",
                    )
                    segmenter = gr.Radio(
                        choices=["sam3", "sapiens", "segformer"],
                        value="sam3",
                        label="Сегментатор (с автоматическим fallback)",
                        info="sam3 — Meta SAM 3 / 3.1 через официальный пакет, "
                             "open-vocabulary, промпт «hair»; "
                             "sapiens — резерв на Sapiens; segformer — лёгкий резерв. "
                             "Падение → каскадный переход.",
                    )
                    sam3_version = gr.Radio(
                        choices=["sam3.1", "sam3"],
                        value="sam3.1",
                        label="Версия SAM 3",
                        info="sam3.1 (март 2026) — Object Multiplex, ~7× быстрее "
                             "на видео, лучше VOS; sam3 (ноябрь 2025) — оригинал. "
                             "На фото качество практически одинаковое.",
                    )
                    sapiens_size = gr.Dropdown(
                        choices=list(SAPIENS_CHECKPOINTS.keys()),
                        value="2b",
                        label="Размер модели Sapiens (используется в fallback)",
                        info="Срабатывает только если SAM 3 не загрузится. "
                             "2b — максимум точности (~8 ГБ); 1b — баланс (~4 ГБ).",
                    )
                    matter = gr.Radio(
                        choices=["matanyone", "feather"],
                        value="matanyone",
                        label="Матирование границ",
                        info="matanyone — нейросетевой alpha-matte по отдельным прядям; "
                             "feather — простое размытие маски (быстрее, но грубее).",
                    )
                    feather_radius = gr.Slider(
                        0, 20, value=4, step=0.5,
                        label="Feather радиус",
                        info="Используется только в режиме feather. На matanyone не влияет.",
                    )
                    device_choice = gr.Radio(
                        choices=["auto", "cuda", "mps", "cpu"],
                        value="cpu",
                        label="Устройство",
                        info="Для SAM 3 / 3.1 на Mac жёстко нужен cpu — Metal "
                             "падает на mixed-dtype matmul внутри модели. "
                             "Sapiens / SegFormer работают на mps. "
                             "auto: CUDA → MPS → CPU.",
                    )
                    do_clean = gr.Checkbox(
                        value=True,
                        label="Чистить маску от шума",
                        info="Удаляет мелкие изолированные пиксели и закрывает дыры. "
                             "Обычно стоит оставить включённым.",
                    )

                with gr.Accordion("📖 Подробное описание параметров", open=False):
                    gr.Markdown(PARAM_HELP)

                run_btn = gr.Button("✨ Применить", variant="primary", size="lg")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Результат"):
                        out_image = gr.Image(
                            label="С заблюренными волосами",
                            type="pil", height=520,
                        )
                    with gr.Tab("Бинарная маска"):
                        out_mask = gr.Image(
                            label="Что модель посчитала волосами",
                            type="pil", height=520,
                        )
                    with gr.Tab("Alpha matte"):
                        out_alpha = gr.Image(
                            label="Полупрозрачные края для плавного блюра",
                            type="pil", height=520,
                        )
                info = gr.Markdown()

        gr.Markdown(
            "_Перед первым запуском с SAM 3.1: установить пакет Meta — "
            "`git clone https://github.com/facebookresearch/sam3.git && "
            "cd sam3 && pip install -e .`, затем "
            "`hf download facebook/sam3.1` (нужен одобренный доступ). "
            "MatAnyone: `pip install git+https://github.com/pq-yang/MatAnyone.git`. "
            "Если что-то из этого не установлено — пайплайн каскадно перейдёт "
            "на Sapiens / SegFormer / feather._"
        )

        run_btn.click(
            process,
            inputs=[inp, blur_radius, mode, segmenter, sapiens_size, sam3_version,
                    matter, feather_radius, device_choice, do_clean, check_woman],
            outputs=[out_image, out_mask, out_alpha, info],
        )

    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="Публичная Gradio-ссылка")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    print(f"detected device: {pick_device(None)}")
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
