# capoeira simulator

[English](README.md) · **Português**

Uma máquina consegue aprender capoeira?

> [!WARNING]
> _Ainda no começo, e todo feito no vibe coding. O corpo ainda cai mais do que qualquer capoeirista. Espere mudanças que quebram tudo._

## o que é?

Um corpo no navegador que aprendeu capoeira por tentativa e erro. É um
manequim de 70 kg no [MuJoCo](https://mujoco.org), o motor de física,
compilado para WebAssembly. As articulações têm torque limitado, como
músculos.

Redes neurais controlam o corpo. Elas foram treinadas por reforço (PPO, no
estilo DeepMimic) para imitar um capoeirista em vídeo. Nada é gravado: a cada
0,03 s uma rede olha o corpo e escolhe os alvos das articulações, e a física
decide o resto. Por isso a ginga não tem fim nem emenda, e segue o andamento
do berimbau.

Peça um movimento a qualquer momento. O corpo escolhe a perna e o momento de
sair da ginga (os dois aprendidos no treino), faz o movimento e volta para a
ginga. Um painel ao lado explica cada movimento: como fazer, dicas e fontes.

## o que ele já faz

A ginga se sustenta nos três toques, com o andamento em qualquer ponto de 70
a 120 bpm. Da ginga o corpo faz quatro movimentos e um giro, e anda pela roda:

| Tecla | Movimento | Termina de pé |
| --- | --- | --- |
| <kbd>1</kbd> | Meia-lua de frente | 165 de 180 |
| <kbd>2</kbd> | Joelhada | 172 de 180 |
| <kbd>3</kbd> | Esquiva lateral | 176 de 180 |
| <kbd>4</kbd> | Cocorinha | 169 de 180 |
| <kbd>5</kbd> | Giro de base | 180 de 180 |

São 60 tentativas em cada toque. Os números vêm do
`scripts/measure-moves.ts`, que roda a mesma simulação do navegador sem
desenhar nada. Um movimento entra no app quando termina de pé nove vezes em
dez e parece o movimento de verdade.

Outras teclas: <kbd>espaço</kbd> toca ou pausa, <kbd>.</kbd> um passo de
controle, <kbd>-</kbd>/<kbd>+</kbd> velocidade, <kbd>WASD</kbd> ou as setas
andam, <kbd>Q</kbd>/<kbd>R</kbd> giram, <kbd>O</kbd> orbita a roda,
<kbd>M</kbd> berimbau. Arraste para girar a câmera. A página abre em inglês,
e as bandeiras trocam para português.

Treinados, mas fora do app até passarem dessa barra:

- **Martelo** (o que veio de vídeo): 134 de 180. Cai no pouso mais ou menos
  uma vez em quatro.
- **Queixada** 147, **troca de base** 142 e **esquiva baixa** 141, de 180.
- **Ponteira** 67 e **bênção** 29, de 90.
- **Meia-lua de compasso**: fica de pé 176 de 180 vezes, mas ainda não é um
  compasso. Sem as mãos no chão, sem o giro.
- **Rolê** 21 e **passagem de negativa** 2, de 90.

## o que ainda não faz

- **Levantar depois de uma queda.** A rede de recuperação levantou 0 de 29
  vezes nos testes ([os números](ml/README-TRAINING.md#resultado-do-treino)).
  Hoje o app põe o corpo de pé de novo depois de 1,2 s.
- **Jogar com alguém.** É um corpo só, sem camarada, então ainda não tem jogo.
- **Trocar de habilidade sem tropeçar.** Ginga, golpes, esquivas e
  recuperação são quatro redes separadas. É na passagem de uma para outra que
  o corpo cai.

## rodando

```bash
npm install
npm run dev        # → http://localhost:3895
npm test           # a simulação contra o Python, os arquivos de movimento, a interpolação, o rig
npm run build
```

O `.envrc` (fora do git) fornece o ambiente (veja `.envrc.example`). Só a
variável `PORT` é usada, e o `vite.config.ts` cai na mesma porta sem ela. O
motor (uns 10 MB) carrega quando a página abre.

## como funciona

```
public/data/sim/ginga/          o modelo, as redes e as referências (ml/export_web.py)
        │  fetch + MuJoCo WASM (src/sim/simulation.ts)
        ▼
LiveGinga                       a cada 0,03 s: observação → rede (ginga, golpes,
                                esquiva ou recuperação) → alvos das articulações → física
        ▼
apply (simulation.ts)           ângulos do MuJoCo → quatérnios dos ossos do rig
        ▼
SkinnedMesh + FloorMarkers      17 ossos, uma malha; prumos e círculos no chão, polígono de apoio
```

- **Estado** (`src/machine.ts`): uma máquina XState cuida do que é discreto:
  tocando ou pausado, velocidade, toque e andamento, o berimbau, a aba do
  painel. A simulação vive fora do React (`src/playback.ts`), então nada
  re-renderiza a cada quadro.
- **Rig** (`src/rig/rig.ts`): uma hierarquia de ossos de verdade ligada a uma
  malha só, com os mesmos ossos do modelo do MuJoCo. Parece um manequim de
  madeira de artista, de calça de abadá branca e corda amarela.
- **Chão** (`src/physics/markers.ts`): cada mão e pé tem um prumo até um
  círculo, preenchido quando toca o chão. Um losango marca o centro de massa
  (segmentos de de Leva, 1996). A linha verde é o polígono de apoio, laranja
  quando o centro de massa sai dele.
- **Triângulo**: o `triangle` em `public/data/movements/ginga.yaml` dá os três
  cantos onde os pés pisam. O treino usa esses cantos, e o app desenha o
  triângulo onde a simulação o pôs: debaixo dos pés depois de um movimento, e
  onde o corpo foi posto de pé depois de uma queda.

### berimbau e andamento

O berimbau é sintetizado no navegador com Tone.js (corda de aço com cabaça
para o *dong* grave e o *ding* agudo, mais o caxixi), em loop, no toque
escolhido:

| Toque | Andamento | Batida (um compasso de 4 tempos) | Ginga |
| --- | --- | --- | --- |
| Angola | 72 bpm | tch-tch DONG ding (pausa) | baixa, larga, fluida, tronco mais inclinado |
| São Bento Pequeno | 90 bpm | tch-tch DING dong (pausa) | média |
| São Bento Grande | 120 bpm | tch-tch DING dong dong | alta, curta e rápida, de pé |

O andamento vai de 70 a 120 bpm, de 2 em 2, e o app começa no São Bento
Pequeno. Uma ginga completa (os dois
lados) dura um compasso, e com a música tocando a fase dela se alinha ao
compasso. A velocidade (0,25× a 1,25×) desacelera música e corpo juntos.
Fontes sobre os toques:
[Capoeira toques](https://en.wikipedia.org/wiki/Capoeira_toques) e
[Capoeira Angola](https://en.wikipedia.org/wiki/Capoeira_Angola) (Wikipedia),
[Ginga (CapoeiraWiki)](https://capoeirawiki.org/wiki/Ginga),
[Angola vs Regional (Abadá Luxemburgo)](https://abadacapoeiraluxemburgo.com/capoeira-angola-capoeira-regional-diferencas/).

## treino

O pipeline fica em [`ml/`](ml/README.md) (Python, uv), com a documentação em
português: vídeo → pose 3D → ângulos das articulações do manequim →
movimentos de referência → PPO → exportação para `public/data/sim/ginga/`. O
treino roda nas GPUs do [Modal](https://modal.com) por padrão, e o mesmo
código roda localmente. Os movimentos estão em
[ml/README-MOVES.md](ml/README-MOVES.md), a recompensa e os resultados em
[ml/README-TRAINING.md](ml/README-TRAINING.md).

## arquivos de movimento

Cada movimento começa num arquivo YAML em `public/data/movements/`: o texto
que o painel mostra (resumo, passos, dicas, fontes) e quadros-chave posados à
mão. O `scripts/sample-clips.ts` amostra esses quadros para o treino. As poses
compartilhadas ficam em `public/data/poses.yaml`. O [RESEARCH.md](RESEARCH.md)
tem os 20 movimentos básicos e as fontes.

```yaml
id: bencao                    # = o nome do arquivo
name: Bênção
category: ataque direto
side: right                   # o lado para o qual os quadros-chave foram escritos
summary: …
steps: [ … ]
keyframes:
  - t: 0                      # segundos
    pose: base_r              # parte de uma pose com nome…
  - t: 0.35
    label: armando o joelho   # …ou do quadro anterior, sobrescrevendo articulações
    hip_r: { flex: 100, abd: 5 }
    knee_r: 105
```

Os ângulos são em graus a partir do repouso: em pé, reto, braços soltos, de
frente para +Z, com o lado esquerdo do corpo para +X. Campo ausente significa
"igual à pose de partida".

| articulação | campos | positivo significa |
|---|---|---|
| `root` | `x`, `z`, `lift` (m) | posição no chão; levantar acima da altura de apoio automática |
| | `yaw`, `pitch`, `roll` | virar à esquerda · inclinar para a frente · inclinar para o próprio lado esquerdo (nessa ordem) |
| `torso`, `head` | `flex`, `side`, `twist` | dobrar para a frente · dobrar para o próprio lado esquerdo · girar à esquerda |
| `shoulder_l/r`, `hip_l/r` | `flex`, `abd`, `rot` | membro para a frente/cima · membro para fora da linha média · rotação interna (abdução primeiro) |
| `elbow_l/r`, `knee_l/r` | número | dobrar |
| `ankle_l/r` | número | apontar os dedos (negativo: dedos para cima) |

O pé fica chapado quando `ankle = hip.flex − knee − root.pitch`. As poses da
ginga inclinam a pelve para a frente (`root.pitch` 10 e 15), então um
quadro-chave que reescreve o `hip.flex` de uma perna de apoio precisa somar
essa inclinação. Um quadro-chave com `pose:` pega a forma da pose, mas mantém
`root.x`, `root.z` e `root.yaw` (uma pose é uma forma, não um lugar).

A folha de contato mostra todos os quadros-chave lado a lado, de vários
ângulos: `/sheet.html?m=martelo` (com `&n=12` para amostras igualmente
espaçadas, `&mirror=1` para o lado esquerdo, `&views=front,top`). Um arquivo
que falha na validação mostra os erros ali e no `npm test`, que também
confere os limites das articulações e que o movimento termina virado para
onde começou, sem sair andando.

## próximos passos

1. **Levantar depois de uma queda.** A rede de recuperação levanta 0 de 29
   vezes, então hoje o app põe o corpo de pé de novo depois de 1,2 s.
2. **Trazer de volta os movimentos que saíram** porque caem demais ou ainda
   não parecem o movimento (a lista está acima).
3. **Mais movimentos de vídeo, nas duas pernas**: queixada e armada, da mesma
   série em câmera lenta do martelo.
4. **Uma política de rastreamento para todos os clipes** em vez de uma rede
   por habilidade. É na passagem entre redes que o corpo cai.
5. **Um guia para adicionar um movimento a partir de um vídeo**, e vídeos de
   capoeiristas. Câmera lenta ou 60 fps, o corpo inteiro no quadro, um
   movimento saindo da ginga e voltando para ela, de frente e de lado.

Tudo isso roda nos US$ 30 de crédito mensal do Modal. Uma H100 custa uns
US$ 0,07 por minuto, e um treino de 40 minutos dá uns 80 M de passos. Fazer
funcionar com esse orçamento faz parte da graça.

## créditos e licenças

- **Referência do martelo**: ajustada a "Martelo -- Capoeira movimento + slow
  motion", do canal [Capoeira Nantes](https://www.youtube.com/watch?v=ZCLVQIfaPHE)
  no YouTube. Nenhuma imagem do vídeo está neste repositório, só os ângulos
  das articulações ajustados a ele (em `public/data/sim/ginga/reference.json`).
  O martelo está treinado, mas fora do app por enquanto.
- **Referência da ginga**: ajustada a um vídeo de um capoeirista, com duas
  câmeras (frente e lado), usando o MediaPipe Pose (Apache-2.0).
- **[GVHMR](https://github.com/zju3dv/GVHMR)** (Shen et al., SIGGRAPH Asia
  2024) transforma o vídeo do martelo em pose 3D. A licença dele é só para
  uso educacional, de pesquisa e sem fins lucrativos; uso comercial precisa
  da permissão dos autores. Ele não está copiado aqui: o `ml/hmr_modal.py` o
  clona quando a imagem do Modal é construída.
- **[SMPL](https://smpl.is.tue.mpg.de) e [SMPL-X](https://smpl-x.is.tue.mpg.de)**:
  os modelos de corpo de que o GVHMR precisa. Pedem cadastro, têm as próprias
  licenças de pesquisa não comercial e não estão neste repositório.
- **[MuJoCo](https://github.com/google-deepmind/mujoco)** (Apache-2.0, Google
  DeepMind), incluindo a versão em WebAssembly, `@mujoco/mujoco`.
- **Artigos**: DeepMimic (Peng et al. 2018) para a recompensa e a imitação,
  KungfuBot (Xie et al. 2025) para as tolerâncias adaptativas, Abdolhosseini
  et al. 2019 para a perda de simetria, de Leva 1996 para as massas dos
  segmentos.
- **Movimentos de quadros-chave** foram posados à mão a partir das fontes do
  [RESEARCH.md](RESEARCH.md).

Ou seja: o código é MIT. As referências de movimento e as redes treinadas
nelas vêm dessa cadeia só para pesquisa e de vídeo de terceiros, então este
projeto não é comercial. Quem quiser usar esses arquivos comercialmente
precisa acertar isso com o GVHMR, o SMPL-X e os donos do vídeo.

## licença

[MIT](LICENSE), para o código.
