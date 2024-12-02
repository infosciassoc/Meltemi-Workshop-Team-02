from llama_index.core.prompts import (
    SelectorPromptTemplate,
    PromptType,
    PromptTemplate,
    MessageRole,
    ChatPromptTemplate,
    ChatMessage,
)

QA_PROMPT = """BEGININPUT
{context_str}

ENDINPUT
BEGININSTRUCTION
{query_str}
ENDINSTRUCTION"""

QA_SYSTEM_PROMPT = """Είσαι ένας ειλικρινής και αμερόληπτος βοηθός που πάντα απαντάει με ακρίβεια σε αυτά που του ζητούνται.
Σου δίνεται ένα έγγραφο το οποίο βρίσκεται μεταξύ του BEGININPUT και του ENDINPUT.
Επίσης, σου δίνονται μεταδεδομένα για το συγκεκριμένο έγγραφο μεταξύ του BEGINCONTEXT και του ENDCONTEXT.
Το βασικό κείμενο του εγγράφου βρίσκεται μεταξύ του ENDCONTEXT και του ENDINPUT.
Απάντα στις οδηγίες του χρήστη που βρίσκονται μεταξύ του BEGININSTRUCTION και του ENDINSTRUCTION χρησιμοποιώντας μόνο το βασικό κείμενο
και τα μεταδεδομένα του εγγράφου που σου δίνονται παρακάτω. Αν οι οδηγίες που σου ζητάει ο χρήστης δεν μπορούν να απαντηθούν με το βασικό
κείμενο ή τα μεταδεδομένα του εγγράφου, ενημέρωσε τον χρήστη ότι δεν ξέρεις την σωστή απάντηση."""

REFINE_PROMPT = """Η αρχική ερώτηση είναι η εξής: {query_str}
Έχουμε δώσει την παρακάτω απάντηση: {existing_answer}
Έχουμε την ευκαιρία να βελτιώσουμε την προηγούμενη απάντηση (μόνο αν χρειάζεται) με τις παρακάτω νέες πληροφορίες.
------------
{context_str}
------------
Με βάση τις νέες πληροφορίες, βελτίωσε την απάντησή σου για να απαντά καλύτερα την ερώτηση.
Αν οι πληροφορίες δεν είναι χρήσιμες, απλά επανέλαβε την προηγούμενη απάντηση.
Βελτιωμένη απάντηση: """

REFINE_CHAT_PROMPT = """Είσαι ένα έμπιστο σύστημα που απαντάει ερωτήσεις χρηστών. Λειτουργείς με τους εξής δύο τρόπους όταν βελτιώνεις υπάρχουσες απαντήσεις:
1. **Ξαναγράφεις** την απάντηση με βάση νέες πληροφορίες που σου παρέχονται
2. **Επαναλαμβάνεις** την προηγούμενη απάντηση αν δεν είναι χρήσιμες οι νέες πληροφορίες

Ποτέ δεν αναφέρεις την αρχική απάντηση ή το συγκείμενο απευθείας στην απάντησή σου.
Αν υπάρχει αμφιβολία για το τι πρέπει να απαντήσεις, απλά επανέλαβε την αρχική απάντηση.
Νέες πληροφορίες:
------------
{context_str}
------------
Ερώτηση: {query_str}
Αρχική απάντηση: {existing_answer}
Βελτιωμένη απάντηση: """


default_template = PromptTemplate(
    metadata={"prompt_type": PromptType.QUESTION_ANSWER},
    # template_vars=["context_str", "query_str"],
    template=QA_PROMPT,
)
chat_template = ChatPromptTemplate(
    metadata={"prompt_type": PromptType.CUSTOM},
    # template_vars=["context_str", "query_str"],
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=QA_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=QA_PROMPT),
    ],
)
text_qa_prompt = SelectorPromptTemplate(
    default_template=default_template, conditionals=[(lambda llm: True, chat_template)]
)

default_refine_template = PromptTemplate(
    metadata={"prompt_type": PromptType.REFINE},
    # template_vars=["query_str", "existing_answer", "context_str"],
    template=REFINE_PROMPT,
)
chat_refine_template = ChatPromptTemplate(
    metadata={"prompt_type": PromptType.CUSTOM},
    # template_vars=["context_str", "query_str", "existing_answer"],
    message_templates=[ChatMessage(role=MessageRole.USER, content=REFINE_CHAT_PROMPT)],
)
text_refine_prompt = SelectorPromptTemplate(
    default_template=default_refine_template,
    conditionals=[(lambda llm: True, chat_refine_template)],
)