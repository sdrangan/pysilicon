# pysilicon/create_hello.py

def create_hello(class_source: str) -> str:
    # naive string manipulation for the toy example
    if "def say_hello" in class_source:
        return class_source  # already added

    insertion = "    def say_hello(self):\n        print('hello world')\n"
    lines = class_source.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("class "):
            # insert after class definition line
            lines.insert(i + 1, insertion)
            break
    return "\n".join(lines)