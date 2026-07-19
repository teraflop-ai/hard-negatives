from datasets import Dataset, Features, Value


def save_text_dataset(
    data,
    *,
    id_column,
    text_column,
    config_name,
    dataset_name,
    hub_path,
    save_path,
    upload: bool = True
):
    dataset = Dataset.from_dict(
        {
            id_column: list(data.keys()),
            text_column: list(data.values()),
        },
        features=Features(
            {
                id_column: Value("int64"),
                text_column: Value("large_string"),
            }
        ),
    )

    dataset.save_to_disk(
        save_path,
    )

    if upload:
        dataset.push_to_hub(
            hub_path,
            config_name=config_name,
            data_dir=config_name,
            split=dataset_name,
        )

        print(f"Pushed {config_name} dataset with {len(dataset)} rows")
