"""Командная строка: python -m reqcheck <требования> <отчёт>"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, dialog, pipeline, report, setup, textout
from .conformance import check_report
from .llm import LLMError, make_client
from .loader import LoadError, load_document


def main(argv: list[str] | None = None) -> int:
    try:  # корректный вывод кириллицы в консоли Windows
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")      # ответы в диалоге --ask, в том числе поданные через пайп
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ap = argparse.ArgumentParser(prog="reqcheck",
                                 description="Агент статической проверки документа с требованиями")
    ap.add_argument("input", help="путь к документу требований (.pdf, .md, .txt, .docx)")
    ap.add_argument("output", nargs="?", default="report.md", help="путь к файлу отчёта: .md или .txt (по умолчанию report.md; "
                                                                  "для .md рядом сохраняется такая же копия в .txt)")
    ap.add_argument("--provider", help="gemini | anthropic (иначе LLM_PROVIDER из .env, по умолчанию gemini)")
    ap.add_argument("--model", help="имя модели (иначе GEMINI_MODEL / ANTHROPIC_MODEL из .env)")
    ap.add_argument("--batch-size", type=int, default=10, help="сколько требований проверять за один вызов (10)")
    ap.add_argument("--workers", type=int, default=3, help="параллельных вызовов модели при проверке требований (3)")
    ap.add_argument("--segmentation", choices=["auto", "rules", "llm"], default="auto",
                    help="нарезка на требования: auto — по разметке, если она есть, иначе моделью; "
                         "rules — только по разметке; llm — только моделью")
    ap.add_argument("--set-passes", type=int, default=2,
                    help="сколько раз независимо повторять проверку набора и поиск противоречий (2): "
                         "объединение выборок снижает разброс между прогонами")
    ap.add_argument("--ask", action="store_true",
                    help="после проверки сразу задать уточняющие вопросы в терминале и учесть ответы автора "
                         "(в живом терминале без этого флага агент спросит, отвечать ли на вопросы; "
                         "без терминала вопросы только записываются в отчёт)")
    ap.add_argument("--no-ask", action="store_true", help="не предлагать диалог с вопросами, даже в живом терминале")
    ap.add_argument("--no-txt", action="store_true", help="не сохранять копию отчёта в .txt рядом с .md")
    ap.add_argument("--no-verify", action="store_true", help="пропустить этап верификации дефектов")
    ap.add_argument("--min-interval", type=float, help="пауза между вызовами модели, с (для бесплатных лимитов)")
    ap.add_argument("--json", metavar="PATH", help="дополнительно сохранить результат в JSON")
    ap.add_argument("--version", action="version", version=f"reqcheck {__version__}")
    args = ap.parse_args(argv)

    try:
        doc = load_document(args.input)
        provider = setup.ensure_credentials(args.provider)     # нет ключа: спросит в терминале и сохранит в .env
        llm = make_client(provider, args.model, args.min_interval)
        print(f"Документ: {doc.name}, строк: {len(doc.lines)}. Модель: {llm.provider}/{llm.model}", file=sys.stderr)
        result = pipeline.run(doc, llm, batch_size=max(args.batch_size, 1),
                              verify=not args.no_verify, workers=max(args.workers, 1),
                              segmentation=args.segmentation, set_passes=max(args.set_passes, 1))
    except (LoadError, LLMError, RuntimeError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        if os.getenv("REQCHECK_DEBUG"):      # полный стек и причина ошибки провайдера: $env:REQCHECK_DEBUG = "1"
            import traceback
            cause = e.__cause__ or e
            traceback.print_exception(type(cause), cause, cause.__traceback__)
            print("тело ответа:", getattr(cause, "body", None), file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    want_dialog = args.ask
    if (not want_dialog and not args.no_ask and result.questions
            and sys.stdin is not None and sys.stdin.isatty() and sys.stdout.isatty()):
        # живой терминал: сначала сохраняем отчёт, затем предлагаем ответить на вопросы сразу
        textout.save_report(out, report.render(result), not args.no_txt)
        want_dialog = dialog.offer(result)
    if want_dialog:
        # Сначала сохраняем отчёт без ответов: если опрос или оценка ответов прервутся, проверка не потеряна.
        textout.save_report(out, report.render(result), not args.no_txt)
        try:
            result, given = dialog.run_dialog(result, llm)
        except (LLMError, RuntimeError) as e:
            print(f"Ошибка при оценке ответов: {e}. Отчёт без ответов сохранён: {out}", file=sys.stderr)
            given = {}
        if given:
            ans_path = out.with_name(out.stem + ".answers.md")
            ans_path.write_text(dialog.answers_markdown(given), encoding="utf-8")
            print(f"Ответы сохранены: {ans_path}", file=sys.stderr)
    md = report.render(result)
    saved = textout.save_report(out, md, not args.no_txt)
    for p in check_report(result, md):
        print(f"Предупреждение (шаблон отчёта): {p}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(report.to_json(result), encoding="utf-8")
    print(f"Готово: {', '.join(str(p) for p in saved)} (требований: {len(result.reqs)}, дефектов: {len(result.active)}, "
          f"вызовов модели: {llm.calls})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
