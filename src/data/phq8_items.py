"""
PHQ-8 item definitions.

This file connects the raw PHQ-8 label columns to the questionnaire item text
that will later be given as input to the model.
"""

PHQ8_ITEMS = {
    1: {
        "item_id": 1,
        "column": "PHQ_8NoInterest",
        "item_text": "Little interest or pleasure in doing things",
        "symptom_name": "NoInterest",
    },
    2: {
        "item_id": 2,
        "column": "PHQ_8Depressed",
        "item_text": "Feeling down, depressed, or hopeless",
        "symptom_name": "Depressed",
    },
    3: {
        "item_id": 3,
        "column": "PHQ_8Sleep",
        "item_text": "Trouble falling or staying asleep, or sleeping too much",
        "symptom_name": "Sleep",
    },
    4: {
        "item_id": 4,
        "column": "PHQ_8Tired",
        "item_text": "Feeling tired or having little energy",
        "symptom_name": "Tired",
    },
    5: {
        "item_id": 5,
        "column": "PHQ_8Appetite",
        "item_text": "Poor appetite or overeating",
        "symptom_name": "Appetite",
    },
    6: {
        "item_id": 6,
        "column": "PHQ_8Failure",
        "item_text": "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
        "symptom_name": "Failure",
    },
    7: {
        "item_id": 7,
        "column": "PHQ_8Concentrating",
        "item_text": "Trouble concentrating on things, such as reading the newspaper or watching television",
        "symptom_name": "Concentrating",
    },
    8: {
        "item_id": 8,
        "column": "PHQ_8Moving",
        "item_text": "Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
        "symptom_name": "Moving",
    },
}


def validate_phq8_items():
    """
    Validate the static PHQ-8 mapping.
    Raises ValueError if the mapping is invalid.
    """
    required_keys = {"item_id", "column", "item_text", "symptom_name"}

    if set(PHQ8_ITEMS.keys()) != set(range(1, 9)):
        raise ValueError("PHQ8_ITEMS must contain exactly item IDs 1-8.")

    for item_id, item in PHQ8_ITEMS.items():
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Item {item_id} is missing required keys: {missing}")

        if item["item_id"] != item_id:
            raise ValueError(f"Item key {item_id} does not match item_id={item['item_id']}.")

        for key in required_keys:
            if item[key] is None or str(item[key]).strip() == "":
                raise ValueError(f"Item {item_id} has empty value for key: {key}")

    return True


if __name__ == "__main__":
    validate_phq8_items()
    print("PHQ8_ITEMS is valid.")