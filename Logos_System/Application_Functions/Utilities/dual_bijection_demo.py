# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

A = {'I': 'C', 'NC': 'T', 'EM': 'T'}
A_inverse = {'C': 'I', 'T': ['NC', 'EM']}

B = {'D': 'E', 'R': 'G', 'A': 'G'}
B_inverse = {'E': 'D', 'G': ['R', 'A']}

def f(x): return A.get(x)
def f_inv(y): return A_inverse.get(y)

def g(x): return B.get(x)
def g_inv(y): return B_inverse.get(y)

# Demo
print(f("I"))       # → 'C'
print(f_inv("T"))   # → ['NC', 'EM']
print(g("R"))       # → 'G'
print(g_inv("G"))   # → ['R', 'A']
