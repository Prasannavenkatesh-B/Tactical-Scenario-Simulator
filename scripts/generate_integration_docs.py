"""CLI utility to regenerate docs/INTEGRATION.md from src/integration/protocol.yaml."""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integration.protocol_docs import generate_integration_md


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    protocol_path = root / "src" / "integration" / "protocol.yaml"
    output_path = root / "docs" / "INTEGRATION.md"

    print(f"[*] Reading protocol specifications from: {protocol_path}")
    generate_integration_md(str(protocol_path), str(output_path))
    print(f"[+] Successfully generated integration guide at: {output_path}")


if __name__ == "__main__":
    main()
