from flask import Flask, render_template, request, jsonify
from main import executar_q_learning

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/treinar', methods=['POST'])
def treinar():
    cfg = request.get_json() or {}
    # parâmetros
    episodios        = int(cfg.get('episodios', 2000))
    taxa_aprendizado = float(cfg.get('alpha',      0.2))
    fator_desconto   = float(cfg.get('gamma',      0.8))
    epsilon          = float(cfg.get('epsilon',    0.9))

    # recompensas
    r_obs            = float(cfg.get('r_obstaculo', -100))
    r_wall           = float(cfg.get('r_parede',    -10))
    r_goal           = float(cfg.get('r_objetivo',  80))

    resultado = executar_q_learning(
        episodios=episodios,
        taxa_aprendizado=taxa_aprendizado,
        fator_desconto=fator_desconto,
        epsilon=epsilon,
        recompensa_obstaculo=r_obs,
        recompensa_parede=r_wall,
        recompensa_objetivo=r_goal
    )
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
