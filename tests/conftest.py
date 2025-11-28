def pytest_addoption(parser):
    parser.addoption(
        "--server-url",
        action="store",
        default=None,
        help="API 서버 URL (예: http://175.209.197.38:8007)"
    ) 