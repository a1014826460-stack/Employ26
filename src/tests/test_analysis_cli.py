from src.analysis import cli


def test_build_parser_accepts_structured_defaults():
    args = cli.build_parser().parse_args(["structured"])

    assert args.command == "structured"
    assert args.structured_command is None
    assert args.with_excel is False
    assert args.skip_standardized is False
    assert args.structured_workers == 4
    assert args.with_legacy_copies is False


def test_build_parser_accepts_structured_run_defaults():
    args = cli.build_parser().parse_args(["structured", "run"])

    assert args.command == "structured"
    assert args.structured_command == "run"


def test_build_parser_accepts_structured_options():
    args = cli.build_parser().parse_args(
        ["structured", "--with-excel", "--structured-workers", "8", "--with-legacy-copies"]
    )

    assert args.command == "structured"
    assert args.with_excel is True
    assert args.structured_workers == 8
    assert args.with_legacy_copies is True


def test_build_parser_accepts_structured_run_options():
    args = cli.build_parser().parse_args(
        ["structured", "run", "--with-excel", "--structured-workers", "6", "--with-legacy-copies"]
    )

    assert args.command == "structured"
    assert args.structured_command == "run"
    assert args.with_excel is True
    assert args.structured_workers == 6
    assert args.with_legacy_copies is True


def test_build_parser_accepts_requirements_options():
    args = cli.build_parser().parse_args(
        ["requirements", "--top-n", "5", "--min-group-size", "2"]
    )

    assert args.command == "requirements"
    assert args.top_n == 5
    assert args.min_group_size == 2


def test_build_parser_accepts_requirements_run_options():
    args = cli.build_parser().parse_args(
        ["requirements", "run", "--top-n", "5", "--min-group-size", "2", "--output-dir", "output/reports/custom_req"]
    )

    assert args.command == "requirements"
    assert args.requirements_command == "run"
    assert args.top_n == 5
    assert args.min_group_size == 2
    assert args.output_dir == "output/reports/custom_req"
