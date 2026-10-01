#!/usr/bin/env python3
"""exe.py — Analizador Léxico de Mini-C.

Implementación ejecutable basada en la especificación de la skill:
/workspaces/PracticaCompiladorC/.agents/skills/analizador-lexico-mini-c/SKILL.md
Especificación elaborada por Alberto Davis (grupo 1SF133), UTP-FISC.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

# Configurar sys.path para importar contratos existentes de minic si están disponibles
CURRENT_DIR = Path(__file__).resolve().parent
for candidate in [CURRENT_DIR / "src", CURRENT_DIR / "CompiladorMiniC-main" / "src"]:
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from minic.lexer.token import Token
    from minic.lexer.token_type import TokenType
    from minic.diagnostics.diagnostic import Diagnostic
    from minic.diagnostics.diagnostic_bag import DiagnosticBag
    from minic.lexer.lexer_result import LexerResult
except ImportError:
    # Definiciones de respaldo idénticas a los contratos del compilador
    class TokenType:
        KW_INT = "KW_INT"
        KW_WHILE = "KW_WHILE"
        IDENTIFIER = "IDENTIFIER"
        INTEGER_LITERAL = "INTEGER_LITERAL"
        ASSIGN = "ASSIGN"
        PLUS = "PLUS"
        MINUS = "MINUS"
        EQUAL_EQUAL = "EQUAL_EQUAL"
        NOT_EQUAL = "NOT_EQUAL"
        LPAREN = "LPAREN"
        RPAREN = "RPAREN"
        LBRACE = "LBRACE"
        RBRACE = "RBRACE"
        SEMICOLON = "SEMICOLON"
        EOF = "EOF"

    @dataclass(frozen=True)
    class Token:
        type: str
        lexeme: str
        literal: int | None
        line: int
        column: int

    @dataclass(frozen=True)
    class Diagnostic:
        code: str
        severity: str
        message: str
        line: int
        column: int

    class DiagnosticBag:
        def __init__(self) -> None:
            self._diagnostics: list[Diagnostic] = []

        def report(self, code: str, message: str, line: int, column: int, severity: str = "error") -> None:
            self._diagnostics.append(Diagnostic(code, severity, message, line, column))

        def has_errors(self) -> bool:
            return any(d.severity == "error" for d in self._diagnostics)

        def __iter__(self):
            return iter(self._diagnostics)

        def __len__(self) -> int:
            return len(self._diagnostics)

        def __getitem__(self, index: int) -> Diagnostic:
            return self._diagnostics[index]

    class LexerResult(NamedTuple):
        tokens: list[Token]
        diagnostics: DiagnosticBag | list[Diagnostic]


# Tablas léxicas según especificación
KEYWORDS: dict[str, str] = {
    "int": TokenType.KW_INT,
    "while": TokenType.KW_WHILE,
}

DOUBLE_OPERATORS: dict[str, str] = {
    "==": TokenType.EQUAL_EQUAL,
    "!=": TokenType.NOT_EQUAL,
}

SINGLE_OPERATORS: dict[str, str] = {
    "=": TokenType.ASSIGN,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ";": TokenType.SEMICOLON,
}

WHITESPACE: frozenset[str] = frozenset({" ", "\t", "\r", "\n"})


def is_alpha(c: str) -> bool:
    return c == "_" or ("a" <= c <= "z") or ("A" <= c <= "Z")


def is_digit(c: str) -> bool:
    return "0" <= c <= "9"


def is_alpha_num(c: str) -> bool:
    return is_alpha(c) or is_digit(c)


class Lexer:
    """Analizador léxico de Mini-C conforme a la especificación de Alberto Davis."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._start = 0
        self._current = 0
        self._line = 1
        self._column = 1
        self._start_line = 1
        self._start_column = 1
        self._tokens: list[Token] = []
        self._diagnostics = DiagnosticBag()

    def scan(self) -> LexerResult:
        """Escanea toda la fuente y devuelve los tokens y diagnósticos acumulados."""
        while not self._is_at_end():
            self._start = self._current
            self._start_line = self._line
            self._start_column = self._column
            self._scan_token()

        # Al terminar la entrada se emite exactamente un token EOF
        eof_token = Token(
            type=TokenType.EOF,
            lexeme="",
            literal=None,
            line=self._line,
            column=self._column,
        )
        self._tokens.append(eof_token)
        return LexerResult(self._tokens, self._diagnostics)

    def _is_at_end(self) -> bool:
        return self._current >= len(self._source)

    def _peek(self) -> str:
        if self._is_at_end():
            return ""
        return self._source[self._current]

    def _peek_next(self) -> str:
        if self._current + 1 >= len(self._source):
            return ""
        return self._source[self._current + 1]

    def _advance(self) -> str:
        char = self._source[self._current]
        self._current += 1
        if char == "\n":
            self._line += 1
            self._column = 1
        else:
            self._column += 1
        return char

    def _scan_token(self) -> None:
        c = self._peek()

        # 1. Blancos: espacio, \t, \r, \n se consumen sin emitir token
        if c in WHITESPACE:
            self._advance()
            return

        # 2. Identificador o palabra reservada: [A-Za-z_][A-Za-z0-9_]*
        if is_alpha(c):
            self._scan_identifier()
            return

        # 3. Entero sin signo: [0-9]+
        if is_digit(c):
            self._scan_number()
            return

        # 4. Operadores o delimitadores (probando operadores dobles primero)
        if self._scan_operator():
            return

        # 5. Cualquier otro carácter fuera de Σ genera LEX001
        self._report_unrecognized()

    def _scan_identifier(self) -> None:
        while is_alpha_num(self._peek()):
            self._advance()

        lexeme = self._source[self._start : self._current]
        token_type = KEYWORDS.get(lexeme, TokenType.IDENTIFIER)
        self._add_token(token_type, literal=None)

    def _scan_number(self) -> None:
        while is_digit(self._peek()):
            self._advance()

        lexeme = self._source[self._start : self._current]
        literal = int(lexeme)
        self._add_token(TokenType.INTEGER_LITERAL, literal=literal)

    def _scan_operator(self) -> bool:
        # Máxima coincidencia: Probar operadores dobles primero (==, !=)
        if self._current + 1 < len(self._source):
            two_chars = self._source[self._current : self._current + 2]
            if two_chars in DOUBLE_OPERATORS:
                self._advance()
                self._advance()
                self._add_token(DOUBLE_OPERATORS[two_chars], literal=None)
                return True

        # Probar operadores simples (=, +, -, (, ), {, }, ;)
        one_char = self._peek()
        if one_char in SINGLE_OPERATORS:
            self._advance()
            self._add_token(SINGLE_OPERATORS[one_char], literal=None)
            return True

        return False

    def _add_token(self, token_type: str, literal: int | None = None) -> None:
        lexeme = self._source[self._start : self._current]
        token = Token(
            type=token_type,
            lexeme=lexeme,
            literal=literal,
            line=self._start_line,
            column=self._start_column,
        )
        self._tokens.append(token)

    def _report_unrecognized(self) -> None:
        bad_char = self._advance()
        msg = f"Carácter no reconocido: '{bad_char}'"
        self._diagnostics.report(
            code="LEX001",
            message=msg,
            line=self._start_line,
            column=self._start_column,
        )


def format_token(token: Token) -> str:
    """Formato: TIPO 'lexema' línea columna."""
    return f"{token.type} '{token.lexeme}' {token.line} {token.column}"


def format_diagnostic(diagnostic: Diagnostic) -> str:
    """Formato: CÓDIGO severidad línea:columna mensaje."""
    return f"{diagnostic.code} {diagnostic.severity} {diagnostic.line}:{diagnostic.column} {diagnostic.message}"


def run_tests() -> bool:
    """Verifica los casos de prueba de la Sección 7 de SKILL.md."""
    print("==================================================")
    print("VERIFICACIÓN DE CASOS DE PRUEBA (SKILL.md §7)")
    print("==================================================")

    # Caso 1: Código válido
    test_1 = "int2 = 12abc;\nwhilex == -5"
    expected_tokens_1 = [
        "IDENTIFIER 'int2' 1 1",
        "ASSIGN '=' 1 6",
        "INTEGER_LITERAL '12' 1 8",
        "IDENTIFIER 'abc' 1 10",
        "SEMICOLON ';' 1 13",
        "IDENTIFIER 'whilex' 2 1",
        "EQUAL_EQUAL '==' 2 8",
        "MINUS '-' 2 11",
        "INTEGER_LITERAL '5' 2 12",
        "EOF '' 2 13",
    ]

    print("\n--- CASO 1: Fuente válida ---")
    print("Entrada:\n" + test_1)
    res_1 = Lexer(test_1).scan()
    actual_tokens_1 = [format_token(t) for t in res_1.tokens]

    print("\nTokens obtenidos:")
    for t_str in actual_tokens_1:
        print(f"  {t_str}")

    ok_1 = (actual_tokens_1 == expected_tokens_1) and (len(res_1.diagnostics) == 0)
    print(f"Resultado Caso 1: {'PASÓ [OK]' if ok_1 else 'FALLÓ [ERROR]'}")

    # Caso 2: Código con errores léxicos
    test_2 = "int x = @;\nx ! = 0; // fin"
    expected_tokens_2 = [
        "KW_INT 'int' 1 1",
        "IDENTIFIER 'x' 1 5",
        "ASSIGN '=' 1 7",
        "SEMICOLON ';' 1 10",
        "IDENTIFIER 'x' 2 1",
        "ASSIGN '=' 2 5",
        "INTEGER_LITERAL '0' 2 7",
        "SEMICOLON ';' 2 8",
        "IDENTIFIER 'fin' 2 13",
        "EOF '' 2 16",
    ]
    expected_diags_2 = [
        "LEX001 error 1:9 Carácter no reconocido: '@'",
        "LEX001 error 2:3 Carácter no reconocido: '!'",
        "LEX001 error 2:10 Carácter no reconocido: '/'",
        "LEX001 error 2:11 Carácter no reconocido: '/'",
    ]

    print("\n--- CASO 2: Fuente con errores léxicos ---")
    print("Entrada:\n" + test_2)
    res_2 = Lexer(test_2).scan()
    actual_tokens_2 = [format_token(t) for t in res_2.tokens]
    actual_diags_2 = [format_diagnostic(d) for d in res_2.diagnostics]

    print("\nTokens obtenidos:")
    for t_str in actual_tokens_2:
        print(f"  {t_str}")

    print("\nDiagnósticos obtenidos:")
    for d_str in actual_diags_2:
        print(f"  {d_str}")

    ok_2_tokens = actual_tokens_2 == expected_tokens_2
    ok_2_diags = actual_diags_2 == expected_diags_2
    ok_2 = ok_2_tokens and ok_2_diags
    print(f"Resultado Caso 2: {'PASÓ [OK]' if ok_2 else 'FALLÓ [ERROR]'}")

    print("\n==================================================")
    all_ok = ok_1 and ok_2
    print(f"ESTADO GENERAL: {'TODAS LAS PRUEBAS PASARON EXITOSAMENTE' if all_ok else 'ALGUNA PRUEBA FALLÓ'}")
    print("==================================================")
    return all_ok


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        if sys.stdin.isatty():
            # Si se ejecuta sin argumentos directamente en terminal, correr pruebas de la especificación
            ok = run_tests()
            return 0 if ok else 1
        # Si recibe entrada por pipe
        source_code = sys.stdin.read()
    elif args[0] in ("-h", "--help"):
        print("Uso:")
        print("  python exe.py                  # Ejecuta las pruebas de la especificación §7")
        print("  python exe.py <archivo.mc>     # Analiza el archivo Mini-C indicado")
        print("  python exe.py -c '<código>'    # Analiza el código pasado como argumento")
        print("  python exe.py --test           # Ejecuta las pruebas de la especificación")
        return 0
    elif args[0] == "--test":
        ok = run_tests()
        return 0 if ok else 1
    elif args[0] == "-c":
        if len(args) < 2:
            print("Error: falta el código tras la opción -c", file=sys.stderr)
            return 2
        source_code = args[1]
    else:
        file_path = Path(args[0])
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                source_code = f.read()
        except OSError as e:
            print(f"Error al leer '{file_path}': {e}", file=sys.stderr)
            return 2

    # Ejecutar análisis léxico
    result = Lexer(source_code).scan()

    if result.diagnostics:
        for diagnostic in result.diagnostics:
            print(format_diagnostic(diagnostic))
        print("La fuente contiene errores léxicos.")
        return 1

    for token in result.tokens:
        print(format_token(token))
    print("Tokens preparados para el analizador sintáctico.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
