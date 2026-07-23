"""Validation chronométrée du modèle Qwen embarqué.

Ce script de diagnostic ne télécharge rien. Il permet de distinguer le temps de
chargement OpenVINO du temps de génération et écrit chaque étape immédiatement,
afin qu'un build ou un diagnostic local puisse détecter un pipeline bloqué.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _event(stage: str, started: float, **details: object) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                **details,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--structured", action="store_true")
    parser.add_argument(
        "--reasoner",
        action="store_true",
        help="Exerce le contrat JSON complet utilisé par l'assistant.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Workspace réel à utiliser pour une question chronométrée.",
    )
    parser.add_argument("--item-id")
    parser.add_argument(
        "--question",
        action="append",
        help="Question réelle ; l'option peut être répétée.",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Modèle absent : {model_path}")

    started = time.perf_counter()
    _event("import", started, device=args.device)
    import openvino_genai

    if args.workspace:
        from transcription_locale.assistant_engine import (
            OpenVINOJsonReasoner,
            propose_speaker_roles,
            verify_claim_result,
        )
        from transcription_locale.audio_dynamics import (
            analyze_audio_dynamics,
            question_needs_audio_analysis,
        )

        workspace = json.loads(
            args.workspace.expanduser().resolve().read_text(encoding="utf-8")
        )
        item_id = args.item_id or next(iter(workspace.get("items", {})), "")
        item = workspace.get("items", {}).get(item_id)
        if not isinstance(item, dict):
            raise SystemExit(f"Fiche audio absente : {item_id}")
        turns = (item.get("result") or {}).get("turns") or []
        overrides = {
            speaker_id: str(value.get("role") or "")
            for speaker_id, value in (item.get("speaker_roles") or {}).items()
            if isinstance(value, dict) and str(value.get("role") or "").strip()
        }
        questions = [
            str(question).strip()
            for question in (args.question or [])
            if str(question).strip()
        ]
        if not questions:
            raise SystemExit("--question est requis avec --workspace")
        reasoner = OpenVINOJsonReasoner(
            model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        roles = propose_speaker_roles(turns, overrides=overrides)
        observations = None
        for question_index, question in enumerate(questions, start=1):
            if (
                observations is None
                and question_needs_audio_analysis(question)
            ):
                observations = analyze_audio_dynamics(
                    item.get("path") or item.get("source_path") or "",
                    turns,
                )
            question_started = time.perf_counter()
            _event(
                "consultation-start",
                started,
                question_index=question_index,
                question=question,
                turns=len(turns),
                device=args.device,
            )
            result = verify_claim_result(
                turns,
                question,
                roles=roles,
                reasoner=reasoner,
                audio_observations=(
                    observations
                    if question_needs_audio_analysis(question)
                    else None
                ),
            )
            _event(
                "consultation-complete",
                started,
                question_index=question_index,
                question_seconds=round(
                    time.perf_counter() - question_started,
                    3,
                ),
                verdict=result.verdict,
                confidence=result.confidence,
                answer=result.answer or result.explanation,
                analysis_points=list(result.analysis_points),
                nuance=result.nuance,
                citations=[
                    {
                        "id": citation.id,
                        "start": round(citation.start, 2),
                        "end": round(citation.end, 2),
                        "role": citation.role,
                    }
                    for citation in result.citations
                ],
                backend=result.backend,
                warnings=list(result.warnings),
                active_device=reasoner.active_device,
                fallback_used=reasoner.fallback_used,
            )
        return 0

    if args.reasoner:
        from transcription_locale.assistant_engine import (
            CONSULTATION_QA,
            Citation,
            ConsultationRequest,
            OpenVINOJsonReasoner,
        )

        _event("reasoner-start", started, device=args.device)
        reasoner = OpenVINOJsonReasoner(
            model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        result = reasoner.answer(
            ConsultationRequest(
                question="Quel est le sujet de cet échange ?",
                intent=CONSULTATION_QA,
                candidates=(
                    Citation(
                        id="turn-00000",
                        turn_index=0,
                        start=0.0,
                        end=5.0,
                        speaker_id="SPEAKER_00",
                        speaker="Interlocuteur 1",
                        role="Master",
                        text="Nous parlons de votre relation avec Cédric.",
                        relevance=1.0,
                        excerpt="Nous parlons de votre relation avec Cédric.",
                    ),
                ),
                role_mapping_confirmed=True,
            )
        )
        _event(
            "reasoner-complete",
            started,
            output=str(result)[:1000],
            active_device=reasoner.active_device,
            fallback_used=reasoner.fallback_used,
        )
        return 0

    _event(
        "pipeline-load-start",
        started,
        model=str(model_path),
        device=args.device,
    )
    pipeline = openvino_genai.LLMPipeline(str(model_path), args.device)
    _event("pipeline-loaded", started, device=args.device)

    config_options: dict[str, object] = {
        "max_new_tokens": max(4, int(args.max_new_tokens)),
        "do_sample": False,
    }
    if args.structured:
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
        config_options["structured_output_config"] = (
            openvino_genai.StructuredOutputConfig(
                json_schema=json.dumps(schema, separators=(",", ":"))
            )
        )
    config = openvino_genai.GenerationConfig(**config_options)
    prompt = (
        "Réponds uniquement en JSON avec une clé answer. "
        'Question: combien font deux plus deux ?\n/no_think'
    )
    _event("generation-start", started, structured=args.structured)
    result = pipeline.generate(prompt, generation_config=config)
    rendered = (
        result
        if isinstance(result, str)
        else getattr(result, "texts", [str(result)])[0]
    )
    _event(
        "generation-complete",
        started,
        output=str(rendered)[:500],
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
