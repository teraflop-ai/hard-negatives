import os
import json


def create_report(
    args,
    dataset_name,
    scores_rows,
    filtered_count,
):
    # Prepare report data
    report_data = {
        "dataset_name": dataset_name,
        "original_rows": len(scores_rows),
        "rows_after_filtering": filtered_count,
        "rows_dropped": len(scores_rows) - filtered_count,
        "retention_rate": filtered_count / len(scores_rows) * 100
        if len(scores_rows) > 0
        else 0,
        "nvembed_threshold": args.nvembed_threshold,
        "min_negatives_required": args.max_negatives_filter,
    }

    # Save report to JSON
    os.makedirs(args.report_output_dir, exist_ok=True)
    report_path = os.path.join(
        args.report_output_dir, f"{dataset_name}_filtering_report.json"
    )
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"Original rows: {report_data['original_rows']}")
    print(
        f"Rows after filtering (>= {args.max_negatives_filter} negatives): {report_data['rows_after_filtering']}"
    )
    print(f"Rows dropped: {report_data['rows_dropped']}")
    print(f"Retention rate: {report_data['retention_rate']:.2f}%")
    print(f"Threshold: {report_data['nvembed_threshold']}")
    print(
        f"Min negatives required: {report_data['min_negatives_required'] if report_data['min_negatives_required'] else 'None (keep all)'}"
    )
    print(f"Report saved to: {report_path}")
    print("=" * 50 + "\n")
