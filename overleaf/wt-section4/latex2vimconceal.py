#!/usr/bin/env python3
"""
latex2vimconceal.py — Generate Vim conceal syntax rules for custom LaTeX macros.

Parses .tex and .sty files for \\newcommand, \\renewcommand, \\providecommand,
\\def, and \\DeclareMathOperator definitions, then emits Vim syntax rules that
conceal them when conceallevel >= 2 (as used by VimTeX).

Usage:
    python3 latex2vimconceal.py main.tex icml2025.sty
    python3 latex2vimconceal.py main.tex icml2025.sty -o conceal.vim
    python3 latex2vimconceal.py main.tex icml2025.sty --inject
    python3 latex2vimconceal.py main.tex icml2025.sty --inject --dry-run
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Unicode lookup tables
# ═══════════════════════════════════════════════════════════════════════════════

MATHBB = {
    'A': '\U0001D538', 'B': '\U0001D539', 'C': '\u2102',    'D': '\U0001D53B',
    'E': '\U0001D53C', 'F': '\U0001D53D', 'G': '\U0001D53E', 'H': '\u210D',
    'I': '\U0001D540', 'J': '\U0001D541', 'K': '\U0001D542', 'L': '\U0001D543',
    'M': '\U0001D544', 'N': '\u2115',     'O': '\U0001D546', 'P': '\u2119',
    'Q': '\u211A',     'R': '\u211D',     'S': '\U0001D54A', 'T': '\U0001D54B',
    'U': '\U0001D54C', 'V': '\U0001D54D', 'W': '\U0001D54E', 'X': '\U0001D54F',
    'Y': '\U0001D550', 'Z': '\u2124',
    '1': '\U0001D7D9',  # blackboard bold 1 (𝟙)
}

MATHCAL = {
    'A': '\U0001D49C', 'B': '\u212C',     'C': '\U0001D49E', 'D': '\U0001D49F',
    'E': '\u2130',     'F': '\u2131',     'G': '\U0001D4A2', 'H': '\u210B',
    'I': '\u2110',     'J': '\U0001D4A5', 'K': '\U0001D4A6', 'L': '\u2112',
    'M': '\u2133',     'N': '\U0001D4A9', 'O': '\U0001D4AA', 'P': '\U0001D4AB',
    'Q': '\U0001D4AC', 'R': '\u211B',     'S': '\U0001D4AE', 'T': '\U0001D4AF',
    'U': '\U0001D4B0', 'V': '\U0001D4B1', 'W': '\U0001D4B2', 'X': '\U0001D4B3',
    'Y': '\U0001D4B4', 'Z': '\U0001D4B5',
}

MATHFRAK = {
    'A': '\U0001D504', 'B': '\U0001D505', 'C': '\u212D',     'D': '\U0001D507',
    'E': '\U0001D508', 'F': '\U0001D509', 'G': '\U0001D50A', 'H': '\u210C',
    'I': '\u2111',     'J': '\U0001D50D', 'K': '\U0001D50E', 'L': '\U0001D50F',
    'M': '\U0001D510', 'N': '\U0001D511', 'O': '\U0001D512', 'P': '\U0001D513',
    'Q': '\U0001D514', 'R': '\u211C',     'S': '\U0001D516', 'T': '\U0001D517',
    'U': '\U0001D518', 'V': '\U0001D519', 'W': '\U0001D51A', 'X': '\U0001D51B',
    'Y': '\U0001D51C', 'Z': '\U0001D51D',
    'a': '\U0001D51E', 'b': '\U0001D51F', 'c': '\U0001D520', 'd': '\U0001D521',
    'e': '\U0001D522', 'f': '\U0001D523', 'g': '\U0001D524', 'h': '\U0001D525',
    'i': '\U0001D526', 'j': '\U0001D527', 'k': '\U0001D528', 'l': '\U0001D529',
    'm': '\U0001D52A', 'n': '\U0001D52B', 'o': '\U0001D52C', 'p': '\U0001D52D',
    'q': '\U0001D52E', 'r': '\U0001D52F', 's': '\U0001D530', 't': '\U0001D531',
    'u': '\U0001D532', 'v': '\U0001D533', 'w': '\U0001D534', 'x': '\U0001D535',
    'y': '\U0001D536', 'z': '\U0001D537',
}

GREEK = {
    'alpha': '\u03B1', 'beta': '\u03B2', 'gamma': '\u03B3', 'delta': '\u03B4',
    'epsilon': '\u03B5', 'varepsilon': '\u03B5', 'zeta': '\u03B6', 'eta': '\u03B7',
    'theta': '\u03B8', 'vartheta': '\u03D1', 'iota': '\u03B9', 'kappa': '\u03BA',
    'lambda': '\u03BB', 'mu': '\u03BC', 'nu': '\u03BD', 'xi': '\u03BE',
    'pi': '\u03C0', 'varpi': '\u03D6', 'rho': '\u03C1', 'varrho': '\u03F1',
    'sigma': '\u03C3', 'varsigma': '\u03C2', 'tau': '\u03C4', 'upsilon': '\u03C5',
    'phi': '\u03D5', 'varphi': '\u03C6', 'chi': '\u03C7', 'psi': '\u03C8',
    'omega': '\u03C9',
    'Gamma': '\u0393', 'Delta': '\u0394', 'Theta': '\u0398', 'Lambda': '\u039B',
    'Xi': '\u039E', 'Pi': '\u03A0', 'Sigma': '\u03A3', 'Upsilon': '\u03A5',
    'Phi': '\u03A6', 'Psi': '\u03A8', 'Omega': '\u03A9',
}

# Direct expression-to-unicode mappings for known patterns in macro expansions
EXPR_TO_UNICODE = {
    r'\mathbb{E}':    '\U0001D53C',  # 𝔼
    r'\mathbb{R}':    '\u211D',      # ℝ
    r'\mathbb{N}':    '\u2115',      # ℕ
    r'\mathbb{Z}':    '\u2124',      # ℤ
    r'\mathbb{C}':    '\u2102',      # ℂ
    r'\mathbb{P}':    '\u2119',      # ℙ
    r'\mathcal{N}':   '\U0001D4A9',  # 𝒩
    r'\mathcal{V}':   '\U0001D4B1',  # 𝒱
    r'\mathcal{D}':   '\U0001D49F',  # 𝒟
    r'\mathcal{L}':   '\u2112',      # ℒ
    r'\mathcal{O}':   '\U0001D4AA',  # 𝒪
    r'\mathcal{F}':   '\u2131',      # ℱ
    r'\mathcal{H}':   '\u210B',      # ℋ
    r'\mathbbm{1}':   '\U0001D7D9',  # 𝟙
    r'\triangleq':    '\u225C',      # ≜
    r'\coloneqq':     '\u2254',      # ≔
    r'\eqqcolon':     '\u2255',      # ≕
    r'\stackrel{\textnormal{\tiny def}}{=}': '\u225D',  # ≝ (equal to by definition)
    r'\mathrel{\stackrel{\textnormal{\tiny def}}{=}}': '\u225D',  # ≝
    r'\operatorname{softmax}': '\u03C3',  # σ (softmax)
}

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Macro:
    """Parsed LaTeX macro."""
    name: str           # without backslash, e.g. 'E'
    nargs: int          # number of arguments
    noptargs: int       # number of optional arguments (usually 0 or 1)
    expansion: str      # raw expansion body
    source_file: str    # which file defined it
    kind: str = ''      # 'newcommand', 'def', 'declaremathop'


@dataclass
class ConcealRule:
    """A classified conceal rule ready for Vim syntax output."""
    category: str       # A, B, C, D, E, F
    macro_name: str     # without backslash
    nargs: int
    noptargs: int
    cchar: Optional[str]  # single Unicode char or None
    vim_lines: list = field(default_factory=list)  # list of vim syntax strings
    comment: str = ''   # optional comment for the rule


# ═══════════════════════════════════════════════════════════════════════════════
# MacroParser
# ═══════════════════════════════════════════════════════════════════════════════

class MacroParser:
    """Parse LaTeX files for macro definitions."""

    # Match \newcommand{\name}[nargs]{body} and variants
    # Also handles \newcommand*
    RE_NEWCMD = re.compile(
        r'\\(?:new|renew|provide)command\*?\s*'
        r'\{?\s*\\([A-Za-z@]+)\s*\}?'        # macro name
        r'(?:\s*\[(\d+)\])?'                   # optional [nargs]
        r'(?:\s*\[([^\]]*)\])?'                # optional [default] for opt arg
        r'\s*\{',                               # opening brace of body
        re.MULTILINE
    )

    # Match \def\name#1#2{body} or \long\def\name#1{body}
    RE_DEF = re.compile(
        r'\\(?:long\s*)?\\?def\s*\\([A-Za-z@]+)'  # macro name
        r'((?:#\d)*)'                               # parameter text like #1#2
        r'\s*\{',                                    # opening brace
        re.MULTILINE
    )

    # Match \DeclareMathOperator{\name}{text} and \DeclareMathOperator*
    RE_DECLMATH = re.compile(
        r'\\DeclareMathOperator\*?\s*'
        r'\{?\s*\\([A-Za-z@]+)\s*\}?'
        r'\s*\{([^}]*)\}',
        re.MULTILINE
    )

    @staticmethod
    def _strip_comments(text: str) -> str:
        """Remove % comments (but not \\%)."""
        lines = text.split('\n')
        out = []
        for line in lines:
            # Remove comments that aren't escaped
            cleaned = re.sub(r'(?<!\\)%.*$', '', line)
            out.append(cleaned)
        return '\n'.join(out)

    @staticmethod
    def _find_matching_brace(text: str, start: int) -> int:
        """Find the matching closing brace, handling nesting.
        `start` should point to the character after the opening brace."""
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{' and (i == 0 or text[i-1] != '\\'):
                depth += 1
            elif text[i] == '}' and (i == 0 or text[i-1] != '\\'):
                depth -= 1
            i += 1
        return i - 1 if depth == 0 else -1

    def parse_file(self, filepath: str) -> list:
        """Parse a single .tex or .sty file for macro definitions."""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()

        text = self._strip_comments(raw)
        macros = []
        fname = os.path.basename(filepath)

        # Parse \newcommand / \renewcommand / \providecommand
        for m in self.RE_NEWCMD.finditer(text):
            name = m.group(1)
            nargs = int(m.group(2)) if m.group(2) else 0
            has_optarg = m.group(3) is not None
            noptargs = 1 if has_optarg else 0
            body_start = m.end()  # just after the opening {
            body_end = self._find_matching_brace(text, body_start)
            if body_end < 0:
                continue
            expansion = text[body_start:body_end].strip()
            macros.append(Macro(
                name=name, nargs=nargs, noptargs=noptargs,
                expansion=expansion, source_file=fname, kind='newcommand'
            ))

        # Parse \def\name#1#2{body}
        for m in self.RE_DEF.finditer(text):
            name = m.group(1)
            params = m.group(2)
            nargs = len(re.findall(r'#\d', params))
            body_start = m.end()
            body_end = self._find_matching_brace(text, body_start)
            if body_end < 0:
                continue
            expansion = text[body_start:body_end].strip()
            # Skip internal LaTeX macros
            if '@' in name:
                continue
            macros.append(Macro(
                name=name, nargs=nargs, noptargs=0,
                expansion=expansion, source_file=fname, kind='def'
            ))

        # Parse \DeclareMathOperator
        for m in self.RE_DECLMATH.finditer(text):
            name = m.group(1)
            optext = m.group(2).strip()
            macros.append(Macro(
                name=name, nargs=0, noptargs=0,
                expansion=rf'\operatorname{{{optext}}}',
                source_file=fname, kind='declaremathop'
            ))

        return macros

    def parse_files(self, filepaths: list) -> list:
        """Parse multiple files, deduplicating by name (last definition wins)."""
        seen = {}
        for fp in filepaths:
            for macro in self.parse_file(fp):
                seen[macro.name] = macro
        return list(seen.values())


# ═══════════════════════════════════════════════════════════════════════════════
# UnicodeMapper
# ═══════════════════════════════════════════════════════════════════════════════

class UnicodeMapper:
    """Map a macro expansion (or macro name) to the best single Unicode char."""

    @staticmethod
    def map_expansion(name: str, expansion: str) -> Optional[str]:
        """Try to find a single-char Unicode conceal for a macro.
        Returns the character or None."""

        # 1. Direct match in EXPR_TO_UNICODE
        for pattern, char in EXPR_TO_UNICODE.items():
            if pattern in expansion:
                return char

        # 2. \mathbb{X}
        m = re.search(r'\\mathbb\{([A-Za-z0-9])\}', expansion)
        if m:
            ch = MATHBB.get(m.group(1))
            if ch:
                return ch

        # 3. \mathbbm{X} (e.g., \mathbbm{1})
        m = re.search(r'\\mathbbm\{([A-Za-z0-9])\}', expansion)
        if m:
            ch = MATHBB.get(m.group(1))
            if ch:
                return ch

        # 4. \mathcal{X}
        m = re.search(r'\\mathcal\{([A-Za-z])\}', expansion)
        if m:
            ch = MATHCAL.get(m.group(1))
            if ch:
                return ch

        # 5. \mathfrak{X}
        m = re.search(r'\\mathfrak\{([A-Za-z])\}', expansion)
        if m:
            ch = MATHFRAK.get(m.group(1))
            if ch:
                return ch

        # 6. Check if expansion contains a Greek letter command
        for greek_name, greek_char in GREEK.items():
            if re.search(r'\\' + greek_name + r'(?![A-Za-z])', expansion):
                return greek_char

        # 7. Macro name itself is a Greek letter name
        if name in GREEK:
            return GREEK[name]

        # 8. \mathrm{X} / \operatorname{X} / \textsc{X} → first letter uppercase
        m = re.search(
            r'\\(?:mathrm|operatorname|textsc)\{([A-Za-z][A-Za-z\s,\\]*)\}',
            expansion
        )
        if m:
            text = m.group(1).strip()
            # Remove LaTeX commands from text to get raw letters
            clean = re.sub(r'\\[A-Za-z]+', '', text).strip()
            clean = re.sub(r'[{}\s,]', '', clean)
            if clean:
                return clean[0].upper()

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MacroClassifier
# ═══════════════════════════════════════════════════════════════════════════════

# Hardcoded overrides: macros that should be treated as comments (fully hidden)
COMMENT_MACROS = {'matan', 'yoav', 'yanai', 'ye', 'comment'}

# Hardcoded overrides: macros that should use emphasis/region styling
# Maps macro name -> (style, color)
# style: 'bold', 'italic', 'smallcaps', None
# color: a color name or None
EMPHASIS_MACROS = {
    'term': ('bold', None),
    'mymacro': (None, 'MacroColor'),
}

# Known macros that VimTeX already conceals — skip these to avoid conflicts
VIMTEX_BUILTINS = {
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta',
    'theta', 'vartheta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi',
    'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma', 'tau', 'upsilon',
    'phi', 'varphi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon',
    'Phi', 'Psi', 'Omega',
    'infty', 'nabla', 'partial', 'forall', 'exists', 'nexists',
    'in', 'notin', 'subset', 'supset', 'subseteq', 'supseteq',
    'cup', 'cap', 'emptyset', 'varnothing',
    'le', 'ge', 'leq', 'geq', 'll', 'gg', 'ne', 'neq', 'approx', 'sim',
    'equiv', 'propto', 'prec', 'succ',
    'pm', 'mp', 'times', 'div', 'cdot', 'circ', 'oplus', 'otimes',
    'to', 'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow',
    'leftrightarrow', 'Leftrightarrow', 'mapsto',
    'langle', 'rangle', 'lceil', 'rceil', 'lfloor', 'rfloor',
    'sum', 'prod', 'int', 'oint', 'bigcup', 'bigcap',
    'sqrt', 'ldots', 'cdots', 'vdots', 'ddots',
    'neg', 'wedge', 'vee', 'top', 'bot', 'perp',
    'triangleq',
    'hat', 'tilde', 'bar', 'vec', 'dot', 'ddot', 'widehat', 'widetilde',
    'textbf', 'textit', 'textsc', 'texttt', 'emph',
    'mathbb', 'mathcal', 'mathfrak', 'mathbf', 'mathrm', 'mathsf',
    'cite', 'citep', 'citet', 'ref', 'eqref', 'label',
    'begin', 'end', 'item', 'section', 'subsection', 'subsubsection',
    'paragraph', 'newcommand', 'renewcommand', 'providecommand',
    'usepackage', 'documentclass', 'input', 'include',
}

# Macros we should skip even if parsed (internal/structural, not user-facing)
SKIP_MACROS = {
    'ICML', 'Notice', 'note', 'icmltitle', 'icmltitlerunning',
    'icmlauthor', 'icmlaffiliation', 'icmlcorrespondingauthor',
    'icmlEqualContribution', 'printAffiliationsAndNotice',
    'icmlkeywords', 'icmladdress', 'icmlsetsymbol',
    'yrcite', 'ftype', 'copyrightspace',
    'abovestrut', 'belowstrut', 'abovespace', 'aroundspace', 'belowspace',
    'texitem', 'icmlitem',
    'addtomylist', 'addstringtofullauthorlist', 'addtofullauthorlist',
    'fnum', 'makecaption',
    # Standard LaTeX formatting/sizing commands
    'small', 'footnotesize', 'scriptsize', 'tiny',
    'large', 'Large', 'LARGE', 'huge', 'Huge',
    'normalsize',
    # Standard LaTeX sectioning/structure (already handled by Vim/VimTeX)
    'thesection', 'thesubsection', 'subparagraph',
    # Internal icml style macros
    'headrulewidth', 'toptitlebar', 'bottomtitlebar',
    'footnoterule', 'addcontentsline', 'icmlruler', 'icmlitem',
    # \cite is already handled by VimTeX
    'cite',
}


class MacroClassifier:
    """Classify macros into conceal categories."""

    @staticmethod
    def should_skip(macro: Macro) -> bool:
        """Return True if this macro should not generate conceal rules."""
        if macro.name in VIMTEX_BUILTINS:
            return True
        if macro.name in SKIP_MACROS:
            return True
        # Skip macros with @ in name (internal LaTeX)
        if '@' in macro.name:
            return True
        # Skip single-letter macros that might conflict with standard LaTeX
        # (except those we explicitly handle)
        return False

    @staticmethod
    def classify(macro: Macro) -> Optional[ConcealRule]:
        """Classify a macro into a category and produce a ConcealRule."""

        if MacroClassifier.should_skip(macro):
            return None

        name = macro.name
        nargs = macro.nargs
        noptargs = macro.noptargs
        expansion = macro.expansion

        # --- Category C: Comment macros (fully hidden including content) ---
        if name in COMMENT_MACROS:
            return ConcealRule(
                category='C', macro_name=name,
                nargs=nargs, noptargs=noptargs, cchar=None,
                comment=f'comment macro from {macro.source_file}'
            )

        # --- Category D: Emphasis macros (delimiters hidden, content styled) ---
        if name in EMPHASIS_MACROS:
            style, color = EMPHASIS_MACROS[name]
            return ConcealRule(
                category='D', macro_name=name,
                nargs=nargs, noptargs=noptargs, cchar=None,
                comment=f'emphasis: style={style}, color={color}'
            )

        # --- Try to find a Unicode mapping ---
        cchar = UnicodeMapper.map_expansion(name, expansion)

        effective_args = nargs - noptargs  # mandatory args count

        if effective_args == 0:
            if cchar:
                # --- Category A: Math symbol with Unicode ---
                return ConcealRule(
                    category='A', macro_name=name,
                    nargs=nargs, noptargs=noptargs, cchar=cchar,
                    comment=f'\\{name} -> {cchar}'
                )
            else:
                # --- Category B: Zero-arg, use first char of name or expansion ---
                fallback = _fallback_char(name, expansion)
                if fallback:
                    return ConcealRule(
                        category='B', macro_name=name,
                        nargs=nargs, noptargs=noptargs, cchar=fallback,
                        comment=f'\\{name} -> {fallback} (fallback)'
                    )
                else:
                    # Completely unmappable, just hide it
                    return ConcealRule(
                        category='B', macro_name=name,
                        nargs=nargs, noptargs=noptargs, cchar=None,
                        comment=f'\\{name} (hidden, no cchar)'
                    )
        else:
            if cchar:
                # --- Category E: Has args + Unicode for the command prefix ---
                return ConcealRule(
                    category='E', macro_name=name,
                    nargs=nargs, noptargs=noptargs, cchar=cchar,
                    comment=f'\\{name}{{...}} -> {cchar}{{...}}'
                )
            else:
                # --- Category F: Has args, no Unicode, just hide the command ---
                fallback = _fallback_char(name, expansion)
                return ConcealRule(
                    category='F', macro_name=name,
                    nargs=nargs, noptargs=noptargs, cchar=fallback,
                    comment=f'\\{name}{{...}} -> {fallback or "hidden"}{{...}}'
                )


def _fallback_char(name: str, expansion: str) -> Optional[str]:
    """Pick a reasonable single fallback character for a macro."""
    # Check for common operator-like patterns
    known_fallbacks = {
        'softmax': '\u03C3',      # σ
        'battnB': 'b',            # b (attention bias, layer 2)
        'modelfull': 'F',         # F for Full model
        'modelminimal': 'M',      # M for Minimal model
        'nbos': 'N',              # N for non-BOS
        # Layer-index notation: use the base letter (lowercase)
        'lidx': '^',              # superscript indicator
        'oB': 'o',               # o_i (attention output)
        'hL': 'h',               # h_i (hidden state)
        'xL': 'x',               # x_i (input)
        'dBOS': 'd',             # d_BOS (direction)
        'dNBOS': 'd',            # d_nonBOS (direction)
        'wBOS': 'w',             # w_BOS (write vector)
        'wNBOS': 'w',            # w_nonBOS (write vector)
    }
    if name in known_fallbacks:
        return known_fallbacks[name]

    # Try to extract meaningful text from expansion
    # \mathrm{text}, \operatorname{text}, \textsc{text}
    m = re.search(
        r'\\(?:mathrm|operatorname|textsc|text)\{([A-Za-z][^}]*)\}',
        expansion
    )
    if m:
        text = m.group(1).strip()
        clean = re.sub(r'\\[A-Za-z]+', '', text)
        clean = re.sub(r'[{}\s\\,]', '', clean)
        if clean:
            return clean[0].upper()

    # \ensuremath{\mathrm{BOS}} pattern
    m = re.search(r'\\mathrm\{([A-Za-z]+)\}', expansion)
    if m:
        return m.group(1)[0].upper()

    # Use first letter of macro name as last resort
    if name and name[0].isalpha():
        return name[0].upper()

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# VimSyntaxGenerator
# ═══════════════════════════════════════════════════════════════════════════════

BEGIN_MARKER = '" ============ BEGIN latex2vimconceal — auto-generated (do not edit manually) ============'
END_MARKER   = '" ============ END latex2vimconceal ============'


class VimSyntaxGenerator:
    """Generate Vim syntax conceal rules from classified macros."""

    def __init__(self, source_names: list):
        self.source_names = source_names

    def _escape_vim(self, name: str) -> str:
        """Escape a macro name for use in Vim regex patterns."""
        # Most macro names are alphanumeric and need no escaping
        return name

    def _cchar_vim(self, ch: Optional[str]) -> str:
        """Format cchar clause for Vim. Only BMP chars work with cchar."""
        if ch is None:
            return ''
        # Check if the character is in the BMP (codepoint <= 0xFFFF)
        # Vim's cchar only supports characters the terminal can display in one cell
        # For supplementary plane chars, we still try — modern terminals handle it
        return f' cchar={ch}'

    def generate_rule_lines(self, rule: ConcealRule) -> list:
        """Generate Vim syntax lines for one rule."""
        lines = []
        name = self._escape_vim(rule.macro_name)

        if rule.category == 'A':
            # Math symbol: 0 args, has Unicode
            cchar = self._cchar_vim(rule.cchar)
            lines.append(
                f"syn match texMathSymbol '\\\\{name}\\>' contained conceal{cchar}"
            )

        elif rule.category == 'B':
            # Zero-arg, hidden or fallback char
            cchar = self._cchar_vim(rule.cchar)
            lines.append(
                f"syn match texStatement '\\\\{name}\\>' contained conceal{cchar}"
            )

        elif rule.category == 'C':
            # Comment region — fully hidden including content
            if rule.noptargs > 0:
                # Has optional arg: \macro[opt]{text}
                lines.append(
                    f"syn region texMyComment matchgroup=texMyCommentDelim "
                    f"start='\\\\{name}\\(\\[.\\{{-}}\\]\\)\\?{{' end='}}' conceal "
                    f"contains=texMyComment containedin=ALL"
                )
            else:
                # No optional arg: \macro{text}
                lines.append(
                    f"syn region texMyComment matchgroup=texMyCommentDelim "
                    f"start='\\\\{name}{{' end='}}' conceal "
                    f"containedin=ALL"
                )

        elif rule.category == 'D':
            # Emphasis region — delimiters hidden, content shown with style
            style, color = EMPHASIS_MACROS.get(rule.macro_name, (None, None))
            group_name = f'texMyConceal{rule.macro_name.capitalize()}'
            border_name = f'{group_name}Border'

            lines.append(
                f"syn region {group_name} matchgroup={border_name} "
                f"start='\\\\{name}{{' end='}}' "
                f"contains=TOP containedin=ALL concealends"
            )
            if style == 'bold':
                lines.append(
                    f"hi def {group_name} term=bold cterm=bold gui=bold"
                )
            elif style == 'italic':
                lines.append(
                    f"hi def {group_name} term=italic cterm=italic gui=italic"
                )
            elif style == 'smallcaps':
                lines.append(
                    f"hi def {group_name} term=bold cterm=bold gui=bold"
                )
            if color:
                lines.append(
                    f"hi def {group_name} ctermfg=DarkGray guifg=#555555"
                )
            lines.append(f"hi def link {border_name} Conceal")

        elif rule.category in ('E', 'F'):
            # Argument macros — command concealed, args visible
            cchar = self._cchar_vim(rule.cchar)
            lines.append(
                f"syn match texStatement '\\\\{name}\\ze{{' contained conceal{cchar}"
            )

        return lines

    def generate(self, rules: list) -> str:
        """Generate the full Vim syntax file content."""
        out = []
        out.append(BEGIN_MARKER)
        out.append(f'" Sources: {", ".join(self.source_names)}')
        out.append(f'" Generated by latex2vimconceal.py')
        out.append('')

        # Group rules by category
        cats = {
            'A': ('Category A: Math symbols (unicode single-char)', []),
            'B': ('Category B: Zero-arg (hidden or fallback)', []),
            'C': ('Category C: Comment regions (fully hidden)', []),
            'D': ('Category D: Emphasis regions (content styled)', []),
            'E': ('Category E: Arg macros (command -> unicode, args visible)', []),
            'F': ('Category F: Arg macros (command hidden, args visible)', []),
        }

        for rule in rules:
            if rule.category in cats:
                cats[rule.category][1].append(rule)

        for cat_key in ('A', 'B', 'C', 'D', 'E', 'F'):
            title, cat_rules = cats[cat_key]
            if not cat_rules:
                continue
            out.append(f'" --- {title} ---')
            for rule in sorted(cat_rules, key=lambda r: r.macro_name):
                if rule.comment:
                    out.append(f'"   {rule.comment}')
                vim_lines = self.generate_rule_lines(rule)
                out.extend(vim_lines)
            out.append('')

        out.append(END_MARKER)
        return '\n'.join(out) + '\n'


# ═══════════════════════════════════════════════════════════════════════════════
# Injection logic
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_INJECT_PATH = os.path.expanduser('~/.vim/after/syntax/tex.vim')


def inject_into_file(content: str, filepath: str, dry_run: bool = False) -> str:
    """Inject or replace the generated block in the target file.
    Returns description of action taken."""
    filepath = os.path.expanduser(filepath)
    parent = os.path.dirname(filepath)

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing = f.read()

        if BEGIN_MARKER in existing and END_MARKER in existing:
            # Replace existing block
            start = existing.index(BEGIN_MARKER)
            end = existing.index(END_MARKER) + len(END_MARKER)
            # Include trailing newline if present
            if end < len(existing) and existing[end] == '\n':
                end += 1
            new_content = existing[:start] + content + existing[end:]
            action = f'Replaced existing block in {filepath}'
        else:
            # Append
            new_content = existing.rstrip('\n') + '\n\n' + content
            action = f'Appended block to {filepath}'
    else:
        # Create new file
        new_content = content
        action = f'Created new file {filepath}'

    if dry_run:
        return f'[DRY-RUN] Would: {action}'

    os.makedirs(parent, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return action


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate Vim conceal syntax rules for custom LaTeX macros.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 latex2vimconceal.py main.tex icml2025.sty
  python3 latex2vimconceal.py main.tex icml2025.sty -o conceal.vim
  python3 latex2vimconceal.py main.tex icml2025.sty --inject
  python3 latex2vimconceal.py main.tex icml2025.sty --inject --dry-run
        """
    )
    parser.add_argument(
        'files', nargs='+',
        help='.tex and .sty files to parse for macro definitions'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Write output to file instead of stdout'
    )
    parser.add_argument(
        '--inject', action='store_true',
        help=f'Inject/update into {DEFAULT_INJECT_PATH}'
    )
    parser.add_argument(
        '--inject-path', default=DEFAULT_INJECT_PATH,
        help=f'Custom injection target path (default: {DEFAULT_INJECT_PATH})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be done without writing any files'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show parsing and classification details'
    )

    args = parser.parse_args()

    # Validate input files
    for fp in args.files:
        if not os.path.isfile(fp):
            print(f'Error: file not found: {fp}', file=sys.stderr)
            sys.exit(1)

    # Parse macros
    mp = MacroParser()
    macros = mp.parse_files(args.files)

    if args.verbose:
        print(f'Parsed {len(macros)} macro definitions:', file=sys.stderr)
        for m in macros:
            print(f'  \\{m.name}[{m.nargs}] = {m.expansion[:60]}... ({m.source_file})',
                  file=sys.stderr)

    # Classify
    rules = []
    skipped = []
    for macro in macros:
        rule = MacroClassifier.classify(macro)
        if rule:
            rules.append(rule)
        else:
            skipped.append(macro.name)

    if args.verbose:
        print(f'\nClassified {len(rules)} rules, skipped {len(skipped)}:',
              file=sys.stderr)
        for s in skipped:
            print(f'  (skipped) \\{s}', file=sys.stderr)
        for r in rules:
            print(f'  [{r.category}] \\{r.macro_name} -> {r.cchar or "(none)"}',
                  file=sys.stderr)

    # Generate
    source_names = [os.path.basename(f) for f in args.files]
    gen = VimSyntaxGenerator(source_names)
    output = gen.generate(rules)

    # Output
    if args.dry_run:
        print(output)
        if args.inject:
            msg = inject_into_file(output, args.inject_path, dry_run=True)
            print(msg, file=sys.stderr)
    elif args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f'Wrote {len(rules)} conceal rules to {args.output}', file=sys.stderr)
    elif args.inject:
        msg = inject_into_file(output, args.inject_path, dry_run=False)
        print(msg, file=sys.stderr)
    else:
        # stdout
        print(output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
