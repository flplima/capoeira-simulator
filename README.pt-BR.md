# capoeira simulator

[English](README.md) · **Português**

Uma máquina consegue aprender capoeira?

https://github.com/user-attachments/assets/dfddc6f9-33ba-4223-b176-dea17ac0d75b

Som ligado! O berimbau também é sintetizado no seu navegador.

Experimente no navegador: **[flplima.github.io/capoeira-simulator](https://flplima.github.io/capoeira-simulator/?lang=pt)**

> [!WARNING]
> _Ainda no começo, e todo feito no vibe coding. O corpo ainda cai mais do que qualquer capoeirista._

## o que é?

Um manequim de 70 kg no [MuJoCo](https://mujoco.org), o motor de física,
rodando no seu navegador. As articulações têm torque limitado, como
músculos. Nada no vídeo é animação.

Redes neurais movem o corpo. Elas aprenderam por tentativa e erro a imitar um
capoeirista em vídeo (aprendizado por reforço, no estilo do
[DeepMimic](https://github.com/xbpeng/DeepMimic)). A cada 0,03 s uma rede olha
o corpo e escolhe os alvos das articulações, e a física faz o resto. Por isso
a ginga segue o berimbau e nunca se repete em loop.

Peça um movimento a qualquer momento. O corpo escolhe a perna e o momento de
sair da ginga, faz o movimento e volta para a ginga.

## o que ele já faz

| Tecla | Movimento | Termina de pé |
| --- | --- | --- |
| <kbd>1</kbd> | Meia-lua de frente | 165 de 180 |
| <kbd>2</kbd> | Joelhada | 172 de 180 |
| <kbd>3</kbd> | Esquiva lateral | 176 de 180 |
| <kbd>4</kbd> | Cocorinha | 169 de 180 |
| <kbd>5</kbd> | Giro de base | 180 de 180 |

São 60 tentativas em cada um dos três toques, de 72 a 120 bpm. Um movimento
entra no app quando termina de pé nove vezes em dez e parece o movimento de
verdade. Outros nove estão treinados e esperam do lado de fora até passarem
dessa barra: a lista e os números estão em
[docs/how-it-works.md](docs/how-it-works.md#moves-still-out-of-the-app) (em
inglês).

## como ele se compara

| | animação por quadros-chave | captura de movimento | capoeira simulator |
|---|---|---|---|
| de onde vem o movimento | um animador posa | a gravação de uma pessoa real | uma rede que aprendeu com um vídeo |
| peso e equilíbrio | desenhados | gravados | simulados: 70 kg, articulações com torque limitado |
| segue o andamento do berimbau | se alguém reajustar | se alguém reajustar | ✓, de 70 a 120 bpm |
| repete em loop | ✓ | ✓ | ✗ |
| pode cair | ✗ | ✗ | ✓ (mais do que devia) |

## rodando

```bash
npm install
npm run dev        # → http://localhost:3895
npm test           # a simulação contra o Python, os arquivos de movimento, o rig
```

A página abre em inglês, e as bandeiras trocam para português. As teclas
ficam embaixo dos botões: <kbd>1</kbd> a <kbd>5</kbd> para os movimentos,
<kbd>WASD</kbd> para andar, <kbd>O</kbd> para circular pela roda.

## como funciona

```
vídeo ──► pose 3D ──► ângulos das articulações do manequim ──► movimentos de referência
                                                                        │
                                          PPO nas GPUs do Modal ◄───────┘
                                                    │
public/data/sim/ginga/ ◄──── exportação ────────────┘
        │
        ▼
o navegador: a cada 0,03 s, estado do corpo ──► rede ──► alvos das articulações ──► MuJoCo
```

O treino fica em [`ml/`](ml/README.md) (Python, documentação em português). O
app é Vite, React e Three.js, e o `src/sim/` roda o mesmo ambiente do
`ml/env.py`, conferido contra ele no `npm test`. O rig, o berimbau, os
arquivos de movimento e o treino estão em
[docs/how-it-works.md](docs/how-it-works.md) (em inglês).

## próximos passos

1. **Levantar depois de uma queda.** A rede de recuperação levanta 0 de 29
   vezes, então hoje o app põe o corpo de pé de novo depois de 1,2 s.
2. **Trazer de volta os movimentos que ficaram de fora**, a começar pelo
   martelo, o que veio de vídeo. Ele cai no pouso mais ou menos uma vez em
   quatro.
3. **Mais movimentos de vídeo, nas duas pernas**: queixada e armada, da mesma
   série em câmera lenta do martelo.
4. **Uma rede para todos os clipes** em vez de uma por habilidade. É na
   passagem entre redes que o corpo cai.
5. **Um guia para adicionar um movimento a partir do seu vídeo.** Câmera
   lenta ou 60 fps, o corpo inteiro no quadro, saindo da ginga e voltando
   para ela.

Tudo isso roda nos US$ 30 de crédito mensal do Modal. Fazer funcionar com
esse orçamento faz parte da graça.

## créditos e licenças

- **Referência do martelo**: ajustada a "Martelo -- Capoeira movimento + slow
  motion", do canal [Capoeira Nantes](https://www.youtube.com/watch?v=ZCLVQIfaPHE)
  no YouTube. Nenhuma imagem do vídeo está neste repositório, só os ângulos
  das articulações ajustados a ele (em `public/data/sim/ginga/reference.json`).
- **Referência da ginga**: ajustada a um vídeo de um capoeirista, com duas
  câmeras (frente e lado), usando o MediaPipe Pose (Apache-2.0).
- **[GVHMR](https://github.com/zju3dv/GVHMR)** (Shen et al., SIGGRAPH Asia
  2024) transforma o vídeo do martelo em pose 3D. A licença dele é só para
  uso educacional, de pesquisa e sem fins lucrativos, e uso comercial precisa
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

## contribuindo

Toda contribuição é bem-vinda, e agentes de IA também! O
[AGENTS.md](AGENTS.md) tem as regras, e o
[docs/CI-TRIAGE.md](docs/CI-TRIAGE.md) mostra o comando local de cada job do
CI.

## licença

[MIT](LICENSE), para o código.
