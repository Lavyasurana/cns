"""Classical ciphers: Caesar, Vigenere, Hill (2x2), and Playfair."""

import math


# 1. CAESAR CIPHER

def caesar_encrypt(plaintext, shift):
    result = ""
    for ch in plaintext:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result += chr((ord(ch) - base + shift) % 26 + base)
        else:
            result += ch
    return result


def caesar_decrypt(ciphertext, shift):
    return caesar_encrypt(ciphertext, -shift)


# 2. VIGENERE CIPHER

def vigenere_encrypt(plaintext, key):
    key = ''.join(c.upper() for c in key if c.isalpha())
    if not key:
        raise ValueError("Key must contain at least one letter.")

    result, j = "", 0
    for ch in plaintext:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            shift = ord(key[j % len(key)]) - ord('A')
            result += chr((ord(ch) - base + shift) % 26 + base)
            j += 1
        else:
            result += ch
    return result


def vigenere_decrypt(ciphertext, key):
    key = ''.join(c.upper() for c in key if c.isalpha())
    if not key:
        raise ValueError("Key must contain at least one letter.")

    result, j = "", 0
    for ch in ciphertext:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            shift = ord(key[j % len(key)]) - ord('A')
            result += chr((ord(ch) - base - shift) % 26 + base)
            j += 1
        else:
            result += ch
    return result


# 3. HILL CIPHER (2x2 KEY MATRIX)

def mod_inverse(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    raise ValueError("Modular inverse does not exist.")


def hill_clean(text):
    return ''.join(c.upper() for c in text if c.isalpha())


def hill_encrypt(plaintext, key):
    if len(key) != 2 or any(len(row) != 2 for row in key):
        raise ValueError("Key must be a 2x2 matrix.")

    a, b = key[0]
    c, d = key[1]
    det = (a * d - b * c) % 26
    if math.gcd(det, 26) != 1:
        raise ValueError("Invalid key: determinant must be coprime with 26.")

    text = hill_clean(plaintext)
    if len(text) % 2:
        text += 'X'

    result = ""
    for i in range(0, len(text), 2):
        x, y = ord(text[i]) - 65, ord(text[i + 1]) - 65
        result += chr((a * x + b * y) % 26 + 65)
        result += chr((c * x + d * y) % 26 + 65)
    return result


def hill_decrypt(ciphertext, key):
    if len(key) != 2 or any(len(row) != 2 for row in key):
        raise ValueError("Key must be a 2x2 matrix.")

    a, b = key[0]
    c, d = key[1]
    det = (a * d - b * c) % 26
    if math.gcd(det, 26) != 1:
        raise ValueError("Invalid key: determinant must be coprime with 26.")

    inv_det = mod_inverse(det, 26)
    inv_key = [
        [(inv_det * d) % 26, (inv_det * -b) % 26],
        [(inv_det * -c) % 26, (inv_det * a) % 26]
    ]

    text = hill_clean(ciphertext)
    if len(text) % 2:
        raise ValueError("Ciphertext must have an even number of letters.")

    result = ""
    for i in range(0, len(text), 2):
        x, y = ord(text[i]) - 65, ord(text[i + 1]) - 65
        result += chr((inv_key[0][0] * x + inv_key[0][1] * y) % 26 + 65)
        result += chr((inv_key[1][0] * x + inv_key[1][1] * y) % 26 + 65)
    return result


# 4. PLAYFAIR CIPHER (I AND J SHARE ONE CELL)

def playfair_matrix(key):
    key = ''.join(c.upper() for c in key if c.isalpha()).replace('J', 'I')
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    chars = []
    for c in key + alphabet:
        if c not in chars:
            chars.append(c)
    return [chars[i:i + 5] for i in range(0, 25, 5)]


def playfair_pairs(text):
    text = ''.join(c.upper() for c in text if c.isalpha()).replace('J', 'I')
    pairs, i = [], 0
    while i < len(text):
        first = text[i]
        if i + 1 == len(text):
            second, i = 'X', i + 1
        elif text[i] == text[i + 1]:
            second, i = 'X', i + 1
        else:
            second, i = text[i + 1], i + 2
        pairs.append(first + second)
    return pairs


def playfair_position(matrix, ch):
    for r in range(5):
        for c in range(5):
            if matrix[r][c] == ch:
                return r, c
    raise ValueError("Character not found in Playfair matrix.")


def playfair_encrypt(plaintext, key):
    matrix = playfair_matrix(key)
    result = ""
    for pair in playfair_pairs(plaintext):
        r1, c1 = playfair_position(matrix, pair[0])
        r2, c2 = playfair_position(matrix, pair[1])

        if r1 == r2:
            result += matrix[r1][(c1 + 1) % 5]
            result += matrix[r2][(c2 + 1) % 5]
        elif c1 == c2:
            result += matrix[(r1 + 1) % 5][c1]
            result += matrix[(r2 + 1) % 5][c2]
        else:
            result += matrix[r1][c2] + matrix[r2][c1]
    return result


def playfair_decrypt(ciphertext, key):
    matrix = playfair_matrix(key)
    text = ''.join(c.upper() for c in ciphertext if c.isalpha()).replace('J', 'I')
    if len(text) % 2:
        raise ValueError("Playfair ciphertext must have an even number of letters.")

    result = ""
    for i in range(0, len(text), 2):
        r1, c1 = playfair_position(matrix, text[i])
        r2, c2 = playfair_position(matrix, text[i + 1])

        if r1 == r2:
            result += matrix[r1][(c1 - 1) % 5]
            result += matrix[r2][(c2 - 1) % 5]
        elif c1 == c2:
            result += matrix[(r1 - 1) % 5][c1]
            result += matrix[(r2 - 1) % 5][c2]
        else:
            result += matrix[r1][c2] + matrix[r2][c1]
    return result


# SAMPLE EXECUTION

if __name__ == "__main__":
    print("CAESAR")
    c = caesar_encrypt("HELLO WORLD", 3)
    print("Encrypted:", c)
    print("Decrypted:", caesar_decrypt(c, 3))

    print("\\nVIGENERE")
    c = vigenere_encrypt("ATTACK AT DAWN", "LEMON")
    print("Encrypted:", c)
    print("Decrypted:", vigenere_decrypt(c, "LEMON"))

    print("\\nHILL")
    key = [[3, 3], [2, 5]]
    c = hill_encrypt("HELP", key)
    print("Encrypted:", c)
    print("Decrypted:", hill_decrypt(c, key))

    print("\\nPLAYFAIR")
    c = playfair_encrypt("HIDE THE GOLD", "MONARCHY")
    print("Encrypted:", c)
    print("Decrypted:", playfair_decrypt(c, "MONARCHY"))
