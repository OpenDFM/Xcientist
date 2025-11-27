class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_section(title: str, char: str = "="):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{char * 80}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{char * 80}{Colors.ENDC}\n")


def print_subsection(title: str):
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}📋 {title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}\n")


def print_success(message: str, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}{Colors.OKGREEN}✓{Colors.ENDC} {message}")


def print_error(message: str, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}{Colors.FAIL}✗{Colors.ENDC} {message}")


def print_warning(message: str, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}{Colors.WARNING}⚠{Colors.ENDC} {message}")


def print_info(message: str, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}{Colors.OKBLUE}ℹ{Colors.ENDC} {message}")


def print_result_box(title: str, content: str, max_length: int = 500):
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}\n")

    if len(content) > max_length:
        display_content = content[:max_length] + "\n... (truncated for display)"
    else:
        display_content = content

    print(f"{Colors.OKGREEN}{display_content}{Colors.ENDC}\n")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}\n")
