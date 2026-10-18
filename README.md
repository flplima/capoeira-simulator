# Capoeira · física real

Um corpo simulado de verdade, ao vivo no navegador, que aprendeu capoeira por
aprendizado de máquina: um boneco sólido no MuJoCo (o motor de física,
compilado para WebAssembly), com esqueleto de manequim e 70 kg, controlado por
redes neurais treinadas por reforço (PPO, no estilo DeepMimic) para imitar a
ginga de um vídeo de referência e os golpes. Nada é gravado: a cada 0,03 s a
rede olha o corpo e decide os próximos alvos das articulações, e a física
decide o resto — a ginga não tem fim nem emenda e segue o andamento do
berimbau.

Os botões **Martelo**, **Bênção** e **Esquiva lateral** (teclas **1**, **2** e
**3**) pedem um movimento a qualquer momento: o corpo escolhe a perna e o
momento de sair da ginga (também aprendidos no treino), golpeia ou esquiva e
volta para a ginga; os botões ficam desativados até ele voltar. Quando perde o
equilíbrio — num golpe que não saiu ou num **Empurrão** (botão ou tecla **E**:
um empurrão de um lado qualquer) — o corpo cai como um capoeirista: para trás,
senta na queda de quatro (mãos e pés no chão); para o lado, desce na negativa,
com a mão no chão e a cabeça protegida; para a frente, apara nas mãos e cai de
joelhos. Depois tenta levantar pelo caminho dessas mesmas posições, voltar para
a base e seguir gingando onde levantou (o triângulo vai junto). Por enquanto a
rede de recuperação ainda quase nunca consegue: o corpo bate a cabeça ou se
perde no chão, e é posto de pé de novo depois de 1,2 s (ver
[`ml/`](ml/README.md#resultado-do-treino)).

O painel ao lado explica cada movimento treinado (ginga, martelo, bênção,
esquiva lateral) — como fazer, dicas e fontes; pedir um movimento abre a aba
dele. O motor (~10 MB) carrega quando a página abre. Código em
`src/sim/simulation.ts`; treino e exportação em [`ml/`](ml/README.md).

Veja [RESEARCH.md](RESEARCH.md) para a pesquisa sobre os movimentos básicos
e as fontes.

## Rodando

```bash
npm install
npm run dev        # → http://localhost:3895
npm test           # a simulação contra o Python, os YAML de referência, a interpolação, o rig
npm run build
```

O `.envrc` (fora do git) fornece o ambiente — veja `.envrc.example`. Só a
variável `PORT` é usada, e o `vite.config.ts` cai na mesma porta sem ela.

Teclas: <kbd>espaço</kbd> tocar/pausar · <kbd>.</kbd> um passo de controle
(0,03 s, pausado) · <kbd>-</kbd>/<kbd>+</kbd> velocidade · <kbd>1</kbd>
<kbd>2</kbd> <kbd>3</kbd> martelo, bênção, esquiva · <kbd>E</kbd> empurrão ·
<kbd>M</kbd> berimbau. Arraste para girar a câmera, role para dar zoom.

## Stack

- Vite + React + TypeScript
- Three.js via @react-three/fiber + @react-three/drei
- MuJoCo 3 em WebAssembly (`@mujoco/mujoco`), redes neurais avaliadas em TypeScript
- XState v5 (+ @xstate/react) para o estado da aplicação
- `yaml` + `zod` para os arquivos de movimento (textos e quadros-chave de referência)
- Vitest

## Como funciona

```
public/data/sim/ginga/          o modelo, as redes e as referências (ml/export_web.py)
        │  fetch + MuJoCo WASM (src/sim/simulation.ts)
        ▼
LiveGinga                       a cada 0,03 s: observação → rede (ginga, golpes, esquiva ou
                                recuperação) → alvos das articulações → física
        ▼
apply (simulation.ts)           ângulos do MuJoCo → quatérnios dos ossos do rig
        ▼
SkinnedMesh + FloorMarkers      17 ossos, uma malha; prumos e círculos no chão, polígono de apoio
```

**Estado (src/machine.ts).** A máquina XState cuida do que é discreto:
`playing | paused`, a velocidade, o toque e o andamento, o berimbau e a aba do
painel. Os controles e o teclado só enviam eventos (`TOGGLE_PLAY`, `STEP`,
`SET_TOQUE`, `SHOW`, …); a simulação vive fora do React (`src/playback.ts`),
então nada re-renderiza a cada quadro.

**Rig (src/rig/rig.ts).** Uma hierarquia de `Bone` de verdade (quadril →
coluna → peito → pescoço → cabeça, braços, pernas, pés) ligada a um único
`SkinnedMesh`, com os mesmos ossos do modelo do MuJoCo. Cada segmento do corpo
é uma cápsula ou esfera com 100 % de peso no seu osso, o que dá a cara de
manequim de madeira de artista: corpo cinza, calça de abadá branca e uma corda
amarela.

**No chão (src/physics/markers.ts).** Cada mão e pé tem um prumo pontilhado até
um pequeno círculo (preenchido quando toca o chão). O losango marca a projeção
do centro de massa (segmentos de de Leva, 1996), e a linha verde é o polígono
de apoio — laranja quando o centro de massa sai dele. Uma grade de 25 cm
(seções de 1 m) cobre o chão.

**Berimbau e andamento.** O botão **♪ Berimbau** toca um berimbau sintetizado
no próprio navegador (Tone.js: corda de aço com ressonância de cabaça para o
*dong* grave e o *ding* agudo, chiado e caxixi), em loop, no toque escolhido:

| Toque | Andamento | Batida (um compasso de 4 tempos) | Ginga |
| --- | --- | --- | --- |
| Angola | 66 bpm | tch-tch DONG ding (pausa) | baixa, larga, fluida, tronco mais inclinado |
| São Bento Pequeno | 90 bpm | tch-tch DING dong (pausa) | média |
| São Bento Grande | 120 bpm | tch-tch DING dong dong | alta, curta e rápida, de pé |

O andamento também pode ser ajustado de 2 em 2 bpm (60–130). Uma ginga
completa (os dois lados) dura um compasso; com a música tocando, a fase da
ginga se alinha ao compasso. A tecla **M** liga e desliga o berimbau, e a
velocidade (0,25×–1,25×) desacelera música e corpo juntos. Fontes sobre os
toques e as diferenças de ginga: [Capoeira toques (Wikipedia)](https://en.wikipedia.org/wiki/Capoeira_toques),
[Capoeira Angola (Wikipedia)](https://en.wikipedia.org/wiki/Capoeira_Angola),
[Ginga — CapoeiraWiki](https://capoeirawiki.org/wiki/Ginga),
[Angola vs Regional (Abadá Luxemburgo)](https://abadacapoeiraluxemburgo.com/capoeira-angola-capoeira-regional-diferencas/).

**Triângulo da ginga.** O `triangle` em `ginga.yaml` dá os três cantos no chão
— os dois da frente são onde os pés ficam na paralela, o de trás é para onde
cada pé recua. O treino usa esses cantos, e o app desenha o triângulo
(tracejado azul; um canto acende quando um pé apoiado está nele) onde a
simulação o pôs: ele vai para debaixo dos pés depois de um golpe e para onde
o corpo levantou depois de uma queda.

## Arquivos de movimento

Os sete arquivos em `public/data/movements/` (ginga, esquiva lateral, martelo,
bênção, queda de quatro, negativa, cocorinha) são o ponto de partida das
referências que o treino imita: `scripts/sample-clips.ts` amostra os
quadros-chave para o `ml/` (ver [`ml/README.md`](ml/README.md)), e o painel do
app mostra os textos dos quatro movimentos treinados.

```yaml
id: bencao                    # = nome do arquivo
name: Bênção
aka: [chapa de frente]
translation: bênção, o gesto de abençoar
category: ataque direto       # fundamento | esquiva | movimentação | ataque circular |
                              # ataque direto | ataque de proximidade | desequilibrante
side: right                   # o lado para o qual os quadros-chave foram escritos
summary: …
steps: [ … ]
tips: [ … ]
sources: [{ title: …, url: … }]
loop: false                   # true + duration para movimentos cíclicos
rest: 0.4                     # pausa antes de repetir (s)
keyframes:
  - t: 0                      # segundos
    pose: base_r              # parte de uma pose com nome…
    label: base
  - t: 0.35
    label: armando o joelho   # …ou do quadro anterior, sobrescrevendo articulações
    hip_r: { flex: 100, abd: 5 }
    knee_r: 105
    torso: { flex: -5 }
```

### Convenções de pose

Todos os ângulos são em graus, relativos à pose de repouso — em pé, reto, braços
soltos, de frente para +Z, com o lado esquerdo do personagem para +X. Campo
ausente significa "igual à pose de partida".

| articulação | campos | positivo significa |
|---|---|---|
| `root` | `x`, `z` (m) | posição no chão |
| | `lift` (m) | levantar acima da altura de apoio automática |
| | `yaw`, `pitch`, `roll` | virar à esquerda · inclinar para a frente · inclinar para o próprio lado esquerdo (aplicados nessa ordem) |
| `torso`, `head` | `flex`, `side`, `twist` | dobrar para a frente · dobrar para o próprio lado esquerdo · girar à esquerda |
| `shoulder_l/r`, `hip_l/r` | `flex`, `abd`, `rot` | membro para a frente/cima · membro para fora da linha média · rotação interna (a abdução é aplicada primeiro, depois a flexão gira esse plano) |
| `elbow_l/r`, `knee_l/r` | número | dobrar |
| `ankle_l/r` | número | apontar os dedos (negativo: dedos para cima) |

O pé fica chapado no chão quando `ankle = hip.flex − knee − root.pitch`.

As poses da ginga (`parallel`, `base_r`, `base_l`) inclinam a pelve para a
frente (`root.pitch` 10 e 15). Um quadro-chave que parte delas herda essa
inclinação; se ele reescreve o `hip.flex` de uma perna de apoio, some o `pitch`
ao ângulo pensado para a pelve reta — senão a perna vai para a frente e o pé
sai do chão.

Um quadro-chave com `pose:` pega a forma do corpo da pose com nome, mas mantém
`root.x`, `root.z` e `root.yaw` do quadro anterior (uma pose é uma forma, não um
lugar). Um quadro sem `pose:` parte do quadro anterior. No `poses.yaml`, uma
entrada pode partir de outra (`pose:`) ou do espelho de outra (`mirror:` — a
`base_l` é `mirror: base_r`).

### Criando e ajustando poses

A folha de contato mostra todos os quadros-chave de um movimento lado a lado,
em várias vistas ortográficas — o jeito mais rápido de conferir uma pose:

```
/sheet.html?m=martelo                         quadros-chave, frente + lado + 3/4
/sheet.html?m=martelo&n=12                    12 amostras igualmente espaçadas (confere os intermediários)
/sheet.html?m=bencao&views=front,top&travel=1 outras vistas; mantém o deslocamento
/sheet.html?m=martelo&mirror=1                a versão do lado esquerdo
```

Um arquivo que falha na validação mostra os erros na folha de contato, num
e no `npm test`, que
também confere os limites das articulações, que o movimento termina virado
para onde começou e que ele não anda para a frente nem para trás.
