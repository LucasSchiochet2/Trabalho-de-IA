import random
import copy
Q_por_ep = []
linhas = 10
colunas = 12
# cima, baixo, esquerda, direita
acoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]

obstaculos = {(1,1),(2,2),(3,1),(2,0),(0,11),(0,4),(1,11),
              (2,6),(5,6),(8,6),(2,8),(2,9),(3,8)}
paredes = {(5,0),(5,1),(5,2),(5,3),(5,8),(5,9),(5,10),(5,11),
           (6,3),(7,3),(8,3),(9,3),(6,8),(7,8),(8,8),(9,8),
    (6,0), (6,1), (6,2), (6,9), (6,10), (6,11),
    (7,0), (7,1), (7,2), (7,9), (7,10), (7,11),
    (8,0), (8,1), (8,2), (8,9), (8,10), (8,11),
    (9,0), (9,1), (9,2), (9,9), (9,10), (9,11)

}

inicio   = (9,4)
objetivo = (4,11)

posicoes_validas = {
    (l, c)
    for l in range(linhas)
    for c in range(colunas)
    if (l,c) not in obstaculos and (l,c) not in paredes
}

def passo(estado, acao, r_obs, r_wall, r_goal):
    l, c = estado
    dl, dc = acoes[acao]
    nl, nc = l + dl, c + dc

    if (nl, nc) in obstaculos:
        return estado, r_obs #bate no obstaculo
    if (nl, nc) in paredes:
        return estado, r_wall #bate na parede
    if (nl, nc) in posicoes_validas:
        if (nl, nc) == objetivo:
            return (nl, nc), r_goal
        return (nl, nc), -1  # movimento normal
    return estado, r_wall  # bate na borda

def escolher_acao(estado, Q, epsilon):
    l, c = estado
    if random.random() < epsilon:
        return random.randrange(4)
    vals = Q[l][c]
    maxv = max(vals)
    melhores = [i for i, v in enumerate(vals) if v == maxv]
    return random.choice(melhores)

def executar_q_learning(
    episodios=300,
    taxa_aprendizado=0.1,
    fator_desconto=0.9,
    epsilon=0.3,
    recompensa_obstaculo=-100,
    recompensa_parede=-10,
    recompensa_objetivo=100
):
    Q = [[[0.0]*4 for _ in range(colunas)] for _ in range(linhas)]
    trajetorias = []

    for _ in range(episodios):
        estado = inicio
        caminho = [estado]
        passos = 0
        Q_por_ep.append(copy.deepcopy(Q))
        while estado != objetivo and passos < 150:
            acao = escolher_acao(estado, Q, epsilon)
            prox, recompensa = passo(
                estado, acao,
                recompensa_obstaculo,
                recompensa_parede,
                recompensa_objetivo
            )

            l, c = estado
            pl, pc = prox

            Q[l][c][acao] += taxa_aprendizado * (
                recompensa +
                fator_desconto * max(Q[pl][pc]) -
                Q[l][c][acao]
            )

            estado = prox
            caminho.append(estado)
            passos += 1

        trajetorias.append(caminho)

    return {"trajetorias": trajetorias, "Q": Q}