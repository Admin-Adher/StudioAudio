from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcription_locale.assistant_engine import (
    CLARIFY,
    CONFIRMED,
    CONTRADICTED,
    NOT_FOUND,
    OpenVINOJsonReasoner,
    PARTIAL,
    ReasonerDecision,
    build_grounded_prompt,
    extract_claim,
    get_default_reasoner,
    propose_speaker_roles,
    retrieve_passages,
    suggest_speaker_roles,
    verify_claim,
    verify_claim_result,
)
from transcription_locale.core import TranscriptTurn


def consultation_turns() -> list[TranscriptTurn]:
    # Le Master parle en second afin de vérifier que l'ordre n'est pas utilisé.
    return [
        TranscriptTurn(
            0.0,
            5.0,
            "VOICE_A",
            "Interlocuteur 1",
            "Bonjour, je voudrais savoir si je vais changer de travail.",
        ),
        TranscriptTurn(
            5.0,
            10.0,
            "VOICE_B",
            "Interlocuteur 2",
            "Donnez-moi votre prénom et votre date de naissance.",
        ),
        TranscriptTurn(
            12.0,
            18.0,
            "VOICE_B",
            "Interlocuteur 2",
            "Je vois que vous allez changer de travail au mois de septembre.",
        ),
        TranscriptTurn(
            18.0,
            21.0,
            "VOICE_A",
            "Interlocuteur 1",
            "D'accord, merci beaucoup.",
        ),
    ]


class RoleProposalTests(unittest.TestCase):
    def test_roles_are_inferred_from_content_not_speaker_order(self) -> None:
        roles = propose_speaker_roles(consultation_turns())

        self.assertEqual(roles.client_speaker_id, "VOICE_A")
        self.assertEqual(roles.master_speaker_id, "VOICE_B")
        self.assertFalse(roles.needs_confirmation)
        self.assertGreaterEqual(roles.confidence, 0.76)

    def test_ambiguous_roles_require_confirmation(self) -> None:
        roles = propose_speaker_roles(
            [
                TranscriptTurn(0, 1, "A", "Voix A", "Bonjour."),
                TranscriptTurn(1, 2, "B", "Voix B", "Bonjour."),
            ]
        )

        self.assertTrue(roles.needs_confirmation)
        self.assertEqual({item.role for item in roles.assignments}, {"Inconnu"})

    def test_one_user_override_completes_a_two_speaker_mapping(self) -> None:
        roles = propose_speaker_roles(
            consultation_turns(),
            overrides={"VOICE_A": "Client"},
        )

        self.assertEqual(roles.role_for("VOICE_A"), "Client")
        self.assertEqual(roles.role_for("VOICE_B"), "Master")
        self.assertEqual(roles.confidence, 1.0)
        self.assertFalse(roles.needs_confirmation)


class RetrievalTests(unittest.TestCase):
    def test_claim_between_quotes_is_extracted(self) -> None:
        question = (
            'La cliente dit que la voyante lui aurait dit "vous allez changer de '
            'travail au mois de septembre". Est-ce vrai ?'
        )
        self.assertEqual(
            extract_claim(question),
            "vous allez changer de travail au mois de septembre",
        )

    def test_natural_direct_question_removes_reporting_frame(self) -> None:
        self.assertEqual(
            extract_claim(
                "est-ce que la master dit que ça va se terminer avec son "
                "conjoint Orion ?"
            ),
            "ça va se terminer avec son conjoint Orion",
        )

    def test_retrieval_returns_original_timecoded_turn(self) -> None:
        roles = propose_speaker_roles(consultation_turns())
        results = retrieve_passages(
            consultation_turns(),
            'La voyante a dit "changer de travail en septembre" ?',
            roles=roles,
        )

        self.assertEqual(results[0].turn_index, 2)
        self.assertEqual(results[0].start, 12.0)
        self.assertEqual(results[0].role, "Master")
        self.assertEqual(
            results[0].text,
            "Je vois que vous allez changer de travail au mois de septembre.",
        )


class VerificationTests(unittest.TestCase):
    def test_exact_claim_is_confirmed_with_real_citation(self) -> None:
        result = verify_claim_result(
            consultation_turns(),
            'La cliente dit que la voyante lui aurait dit "vous allez changer de '
            'travail au mois de septembre". Est-ce vrai ?',
        )

        self.assertEqual(result.verdict, CONFIRMED)
        self.assertEqual(result.target_role, "Master")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].turn_index, 2)
        self.assertIn(result.citations[0].text, consultation_turns()[2].text)
        self.assertTrue(result.answer.startswith("Oui."))
        self.assertIn("changer de travail", result.answer)
        self.assertTrue(result.analysis_points)
        self.assertIn("00:12", result.analysis_points[0])
        self.assertIn("prédiction", result.nuance)

    def test_exact_quote_wins_over_an_unrelated_negation_later_in_the_turn(self) -> None:
        turns = consultation_turns()
        turns[2] = TranscriptTurn(
            12.0,
            25.0,
            "VOICE_B",
            "Interlocuteur 2",
            "Il y a un renouveau possible entre vous et Orion, mais je ne "
            "peux pas préciser si une autre personne interviendra.",
        )

        result = verify_claim_result(
            turns,
            'La voyante a dit "il y a un renouveau possible entre vous et Orion".',
        )

        self.assertEqual(result.verdict, CONFIRMED)
        self.assertEqual(result.citations[0].turn_index, 2)

    def test_partial_claim_is_reported_as_partial(self) -> None:
        result = verify_claim_result(
            consultation_turns(),
            'La voyante aurait dit "vous allez changer de travail en septembre '
            'avec une importante augmentation de salaire".',
        )

        self.assertEqual(result.verdict, PARTIAL)
        self.assertTrue(result.citations)
        self.assertEqual(result.citations[0].turn_index, 2)
        self.assertTrue(result.answer.startswith("Partiellement."))
        self.assertIn("chaque détail", result.answer)

    def test_opposite_polarity_is_reported_as_contradiction(self) -> None:
        turns = consultation_turns()
        turns[2] = TranscriptTurn(
            12.0,
            18.0,
            "VOICE_B",
            "Interlocuteur 2",
            "Je vois que vous ne changerez pas de travail au mois de septembre.",
        )
        result = verify_claim_result(
            turns,
            'La voyante a dit "vous changerez de travail au mois de septembre".',
        )

        self.assertEqual(result.verdict, CONTRADICTED)
        self.assertEqual(result.citations[0].text, turns[2].text)

    def test_relation_end_synonyms_and_local_negation_use_early_evidence(self) -> None:
        turns = [
            TranscriptTurn(
                26.0,
                35.0,
                "CLIENT",
                "Client",
                "Je voudrais savoir ce qui va arriver avec Orion.",
            ),
            TranscriptTurn(
                388.0,
                431.0,
                "MASTER",
                "Master",
                "Pour Orion et Lyra, j'ai des notions de fin des relations. "
                "Les deux personnes se lâchent et se laissent partir chacune "
                "de son côté.",
            ),
            TranscriptTurn(
                431.0,
                432.3,
                "CLIENT",
                "Client",
                "Pour elle, il y a quand même une...",
            ),
            TranscriptTurn(
                432.3,
                483.4,
                "MASTER",
                "Master",
                "Ils vont se lâcher et se laisser partir l'un d'autre, chacun "
                "de son côté. Ils resteront peut-être en contact, mais je ne "
                "vois pas de reprise de relation amoureuse avec Orion.",
            ),
            TranscriptTurn(
                728.2,
                779.8,
                "MASTER",
                "Master",
                "Il est hors de question que vous soyez un second choix. Je "
                "pense au fait de mettre fin à une relation avec Lyra. "
                "Orion n'est pas porté sur le fait de faire des choix.",
            ),
        ]

        result = verify_claim_result(
            turns,
            "est-ce que la master dit que ça va se terminer avec son conjoint "
            "Orion ?",
        )

        self.assertEqual(result.claim, "ça va se terminer avec son conjoint Orion")
        self.assertEqual(result.verdict, CONFIRMED)
        self.assertGreaterEqual(len(result.citations), 2)
        self.assertEqual([item.start for item in result.citations[:2]], [388.0, 432.3])

    def test_ambiguous_partner_question_asks_which_relationship_is_targeted(self) -> None:
        turns = [
            TranscriptTurn(
                0.0,
                5.0,
                "CLIENT",
                "Client",
                "Je voudrais savoir ce qui va arriver avec Orion.",
            ),
            TranscriptTurn(
                10.0,
                18.0,
                "MASTER",
                "Master",
                "Je vois un renouveau possible entre vous et Orion et votre lien va se stabiliser.",
            ),
            TranscriptTurn(
                20.0,
                30.0,
                "MASTER",
                "Master",
                "Pour Orion et Lyra, j'ai des notions de fin de relation et de séparation.",
            ),
        ]

        result = verify_claim_result(
            turns,
            "est-ce que la master dit que ça va se terminer avec son conjoint Orion ?",
        )

        self.assertEqual(result.verdict, CLARIFY)
        self.assertIn("deux relations", result.answer)
        self.assertEqual(len(result.analysis_points), 2)
        self.assertIn("relations différentes", result.nuance)
        self.assertEqual(
            [option["label"] for option in result.clarification_options],
            ["Orion et Lyra", "La cliente et Orion"],
        )
        self.assertEqual([item.start for item in result.citations], [20.0, 10.0])

        refined = verify_claim_result(
            turns,
            "Est-ce que le Master dit que la relation entre Orion et Lyra va se terminer ?",
        )
        self.assertEqual(refined.verdict, CONFIRMED)
        self.assertEqual(refined.citations[0].start, 20.0)

    def test_relation_end_is_contradicted_only_when_predicate_is_negated(self) -> None:
        turns = [
            TranscriptTurn(
                0.0,
                5.0,
                "CLIENT",
                "Client",
                "Je voudrais savoir pour ma relation.",
            ),
            TranscriptTurn(
                5.0,
                12.0,
                "MASTER",
                "Master",
                "Je vous confirme que la relation avec Orion ne se terminera "
                "pas.",
            ),
        ]

        result = verify_claim_result(
            turns,
            "est-ce que la master dit que ça va se terminer avec son conjoint "
            "Orion ?",
        )

        self.assertEqual(result.verdict, CONTRADICTED)
        self.assertEqual(result.citations[0].start, 5.0)

    def test_unrelated_claim_is_not_found_and_has_no_citation(self) -> None:
        result = verify_claim_result(
            consultation_turns(),
            'La voyante a dit "vous achèterez un bateau bleu au Canada".',
        )

        self.assertEqual(result.verdict, NOT_FOUND)
        self.assertEqual(result.citations, ())
        self.assertIn("ne prouve pas", result.explanation)

    def test_mapping_turns_are_supported(self) -> None:
        raw = [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker_id": turn.speaker_id,
                "speaker": turn.speaker,
                "text": turn.text,
            }
            for turn in consultation_turns()
        ]
        result = verify_claim_result(
            raw,
            'La voyante a dit "changer de travail au mois de septembre".',
        )
        self.assertEqual(result.verdict, CONFIRMED)

    def test_invalid_model_citation_is_rejected_and_falls_back(self) -> None:
        class HallucinatingReasoner:
            name = "faux-modèle"

            def decide(self, request):
                return {
                    "verdict": "Confirmé",
                    "confidence": 0.99,
                    "citation_ids": ["citation-inventée"],
                    "quote": "Une phrase qui n'existe pas.",
                }

        result = verify_claim_result(
            consultation_turns(),
            'La voyante a dit "vous achèterez un bateau bleu au Canada".',
            reasoner=HallucinatingReasoner(),
        )

        self.assertEqual(result.verdict, NOT_FOUND)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.backend, "déterministe-local")
        self.assertTrue(result.warnings)
        self.assertNotIn("Une phrase qui n'existe pas", str(result.to_dict()))

    def test_valid_model_can_only_select_existing_passages(self) -> None:
        class GroundedReasoner:
            name = "modèle-local-test"

            def decide(self, request):
                return ReasonerDecision(
                    verdict=CONFIRMED,
                    confidence=0.91,
                    citation_ids=(request.candidates[0].id,),
                )

        result = verify_claim_result(
            consultation_turns(),
            'La voyante a dit "changer de travail en septembre".',
            reasoner=GroundedReasoner(),
        )

        self.assertEqual(result.backend, "modèle-local-test")
        self.assertEqual(result.citations[0].text, consultation_turns()[2].text)

    def test_grounded_prompt_lists_only_retrieved_ids(self) -> None:
        class PromptCapturingReasoner:
            name = "capture"

            def decide(self, request):
                prompt = build_grounded_prompt(request)
                self.prompt = prompt
                return {
                    "verdict": "Confirmé",
                    "confidence": 0.8,
                    "citation_ids": [request.candidates[0].id],
                }

        reasoner = PromptCapturingReasoner()
        verify_claim_result(
            consultation_turns(),
            'La voyante a dit "changer de travail en septembre".',
            reasoner=reasoner,
        )
        self.assertIn("turn-00002", reasoner.prompt)
        self.assertIn("N'invente aucune citation", reasoner.prompt)

    def test_ui_facades_return_json_safe_roles_citations_and_coverage(self) -> None:
        roles = suggest_speaker_roles(consultation_turns())
        payload = verify_claim(
            'La voyante a dit "changer de travail en septembre".',
            consultation_turns(),
            roles,
        )

        self.assertEqual(roles["roles"]["VOICE_A"], "Client")
        self.assertEqual(payload["verdict"], CONFIRMED)
        self.assertEqual(payload["coverage"]["turns_analyzed"], 4)
        self.assertEqual(payload["coverage"]["scope"], "transcription-complete")
        citation = payload["citations"][0]
        self.assertEqual(citation["quote"], consultation_turns()[2].text)
        self.assertEqual(citation["start_seconds"], 12.0)
        self.assertEqual(citation["end_seconds"], 18.0)
        self.assertEqual(citation["speaker_id"], "VOICE_B")
        self.assertEqual(citation["role"], "Master")


class OptionalReasonerTests(unittest.TestCase):
    def test_default_reasoner_is_absent_without_a_local_model(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(get_default_reasoner())

    def test_openvino_reasoner_is_lazy_and_uses_only_grounded_ids(self) -> None:
        calls: list[tuple[str, str]] = []

        class FakePipeline:
            def generate(self, prompt, **kwargs):
                self.prompt = prompt
                return (
                    '```json\n{"verdict":"Confirmé","confidence":0.9,'
                    '"citation_ids":["turn-00002"]}\n```'
                )

        with TemporaryDirectory() as directory:
            reasoner = OpenVINOJsonReasoner(
                Path(directory),
                pipeline_factory=lambda path, device: (
                    calls.append((path, device)) or FakePipeline()
                ),
            )
            self.assertFalse(reasoner.loaded)
            payload = verify_claim(
                'La voyante a dit "changer de travail en septembre".',
                consultation_turns(),
                reasoner=reasoner,
            )

        self.assertTrue(reasoner.loaded)
        self.assertEqual(len(calls), 1)
        self.assertEqual(payload["backend"], "OpenVINO-local")
        self.assertEqual(payload["citations"][0]["quote"], consultation_turns()[2].text)


if __name__ == "__main__":
    unittest.main()
