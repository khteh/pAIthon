from sympy import symbols, diff

def Derivatives1P(J, w):
    print(f"\n=== {Derivatives1P.__name__} ===")
    print(f"type(J): {type(J)}")
    print(f"J: {J}")
    dJ_dw = diff(J, w)
    print(f"type(dJ_dw): {type(dJ_dw)}")
    print(f"dJ_dw: {dJ_dw}")
    result = dJ_dw.subs([(w, 2)])
    print(f"type(result): {type(result)}")
    print(f"w=2: {result}")
    result = dJ_dw.subs([(w, 3)])
    print(f"w=3: {result}")
    result = dJ_dw.subs([(w, -3)])
    print(f"w=-3: {result}")

def Derivatives2P():
    """
    J = (2 + 3w) ** 2
    a = 2 + 3*w
    dJ/dw = dJ/da * da/dw
    """
    print(f"\n=== {Derivatives2P.__name__} ===")
    a = 2 + 3*w
    sw,sJ,sa = symbols('w,J,a')

    sJ = sa**2
    sJ.subs([(sa,a)])
    dJ_da = diff(sJ, sa)
    print(f"dJ_da: {dJ_da}") # 2*a

    sa = 2 + 3*sw
    da_dw = diff(sa, sw)
    print(f"da_dw: {da_dw}") # 3

    dJ_dw = dJ_da * da_dw
    print(f"dJ_dw: {dJ_dw}") # 6*a

if __name__ == "__main__":
    J, w = symbols("J, w")
    J = w ** 2
    Derivatives1P(J, w)
    J = 1 / w
    Derivatives1P(J, w)   
    J = 1 / (w ** 2)
    Derivatives1P(J, w)
    Derivatives2P()