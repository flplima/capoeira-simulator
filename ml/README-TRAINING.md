# Treino, navegador e resultado

Continuação do [pipeline](README.md#pipeline): a recompensa, o treino, o navegador e o resultado do treino.

- **Recompensa (env_reward.py).** Quão perto a simulação fica da referência: ângulos
  (40 %), velocidades (10 %), posição de mãos, pés e cabeça (15 %), da raiz
  (15 %) e de cada pé no seu ponto do triângulo, no mundo (20 %). As mãos,
  pés e cabeça são medidos virados para onde o corpo olha, então durante um
  golpe 15 % da recompensa vai para onde o pé do chute está em relação ao
  quadril sem descontar esse giro (`strike`): um martelo que sai para o lado
  sem o quadril virar perde ali o que ganhava nos outros termos, e 20 % vai
  para o pé de apoio girar como o da referência (`pivot`): a direção do pé, a
  altura do calcanhar e a bola do pé parada onde a da referência está (no
  martelo o pé gira na bola, calcanhar 7 cm acima). Sem o calcanhar e a bola,
  o corpo girava o pé chapado e o arrastava 0,3–0,7 m a cada golpe. Cada
  termo vale exp(−erro²/σ), e a tolerância σ é o próprio treino que põe
  (`env_consts.TERMS`, `env_math.tighten`): a média corrente do erro daquele
  termo, só apertando depois das primeiras iterações (como no KungfuBot, Xie
  et al. 2025; as primeiras, todas de episódios recém-começados, erram pouco
  demais para servir de piso). No começo tudo é
  folgado; conforme o corpo chega mais perto, a tolerância desce atrás dele,
  e um erro que ele nunca fecha — uma referência que é só uma estimativa, de
  uma câmera ou de quadros-chave — fica folgado em vez de custar recompensa
  para sempre. Há um jogo de tolerâncias para a ginga (e o chão) e um para
  cada golpe ou esquiva em cada perna: com um jogo só para todos os golpes, o
  que aprende primeiro aperta a tolerância a um ponto em que os outros pontuam
  zero e param de aprender (o martelo direito chegou a 85 % enquanto o
  esquerdo ficou em 10 %; o martelo do vídeo ficou em zero no meio de golpes
  de quadros-chave a 90 %). A cada tamanho novo do currículo elas voltam a folgar.
  No `log.jsonl` elas aparecem como erro RMS por termo (`tol`), que é a medida
  de progresso: a recompensa em si fica perto de e⁻¹ por construção. O episódio
  acaba quando algo além de mãos e pés toca o chão, a cabeça desce demais ou o
  corpo se afasta 0,5 m da referência; começa em uma fase e um andamento
  aleatórios e dura até 40 s, com o toque mudando de vez em quando (em média
  a cada 8 s), para a política aprender a acompanhar a música quando ela
  muda. Como o triângulo é fixo no mundo, ela aprende a gingar no lugar.
- **Treino (modal_train.py, train_mjx.py, train.py).** PPO com GAE, rede MLP
  de 512–256 para política e valor, observações normalizadas. O padrão é o
  Modal: `modal_train.py` roda o `train_mjx.py` (MuJoCo em JAX, 2048
  simulações em paralelo) numa GPU (L4 por padrão; `--gpu` troca, com
  `--envs` à altura da memória), e os checkpoints ficam no volume
  `capoeira-mjx-runs` até `modal_train.py::fetch` trazê-los para `runs/`.
  No próprio Mac, o `train_mjx.py` roda na CPU (64 simulações) e o `train.py`
  é o PPO original em PyTorch (12 simulações na CPU); os três gravam o mesmo
  formato de `.pt`.
  `--init` parte de um checkpoint anterior (entradas novas começam com peso
  zero). Um golpe só conta como feito de pé depois de 1 s de volta na ginga
  (`KICK_SETTLE`, também no `check.py`): o corpo que acabava um golpe girado
  caía logo depois, com o golpe já contado. O tamanho do golpe sobe quando 70 % termina de pé ou, no máximo, a
  cada 18 % do tempo, e aos 75 % vai ao tamanho inteiro; `--gated` tira o
  relógio e o tamanho só sobe com o acerto, para um golpe novo. `--focus`
  dá metade dos golpes a um movimento, `--only` dá todos. `--mirror` treina
  também com o espelho de cada lote (`env_mirror.py`): a referência da ginga
  é o próprio espelho meio ciclo adiante, o triângulo é simétrico e a perna
  esquerda de cada golpe espelha a direita, então um estado espelhado vale
  tanto quanto o original e a rede deve agir nele espelhada (Abdolhosseini et
  al. 2019). Cada amostra entra duas vezes, a segunda com observação e ação
  espelhadas (x troca de sinal, esquerda e direita trocam, a torção e a
  inclinação lateral do tronco trocam de sinal, a fase da ginga anda meio
  ciclo), com a mesma vantagem: um golpe aprende as duas pernas de uma vez.
  Sem isso o martelo direito chegou a 85 % com o esquerdo em 12 %, e só o
  currículo do tamanho 0 recuperou a esquerda. `uv run python env_mirror.py`
  confere que a observação do estado espelhado é a observação espelhada.
- **No navegador (src/sim/simulation.ts).** O mesmo `humanoid.xml` roda no
  MuJoCo 3.15 compilado para WebAssembly (`@mujoco/mujoco`, a mesma versão do
  Python). A cada 0,03 s o app monta a observação exatamente como o `env.py`,
  passa pela rede (três camadas, 204 mil pesos, em JavaScript puro) e aplica
  os alvos aos motores; 15 passos de física de 2 ms depois, repete. A fase da
  ginga é puxada suavemente para o compasso do berimbau quando a música toca.
  Se o corpo cair, ele é posto de pé de novo depois de 1,2 s. O teste
  `test/live.test.ts` confere observação, ação e estado seguinte contra
  valores calculados em Python.

## Resultado do treino

Primeira rodada (sem triângulo): 110 minutos, 22 milhões de passos; os
episódios foram de 0,8 s a ~17 s e a recompensa de 0,30 a ~0,55, mas o corpo
escorregava até 1,7 m de lado em quatro ciclos.

Segunda rodada (triângulo), partindo da primeira: 120 minutos, 20 milhões de
passos. A recompensa foi de 0,38 a 0,87 (pés no triângulo 0,33 → 0,88). A
política final gingou 60 ciclos seguidos (216 s) sem cair, sem se afastar
mais de 0,2 m do centro do triângulo, com o pé apoiado a ~3 cm do canto
(mediana). O movimento entra num ciclo estável: o estado no fim do trecho
exportado é praticamente igual ao do início, então o loop no app não tem
emenda — é uma ginga sem fim.

Cair e levantar (rede de recuperação), partindo da ginga: três testes de 15
minutos e uma rodada de 4 horas (100 milhões de passos). No treino, a parte
dos episódios que termina de pé subiu de 0,26 para ~0,36 — mas lá muitos
começam no meio do caminho de volta. No teste de 3 minutos (ginga e golpes
das melhores redes, empurrões de 350 N a cada 4 s, golpes a cada 5 s, seis
andamentos de 66 a 120 bpm) o corpo levantou de 0 de 29 quedas (a rodada de
4 h; a anterior, 1 de 30): em ~70 % delas a cabeça toca o chão logo depois
do pouso (0,4–0,9 s após perder o equilíbrio) e o resto se perde no chão. O
corpo não chega à pose de pouso da referência — rola de costas de quatro
apoios, cai de cabeça na negativa. A meta é 80 % de pé com no máximo 10 % de
cabeça no chão; até lá o app põe o corpo de pé de novo. Com empurrões, os
golpes terminam de pé martelo 10 de 13 e bênção 9 de 13 com a rede de
recuperação, e 11 de 18 e 15 de 18 sem ela — diferença dentro do ruído de
amostras tão pequenas. Sem empurrões, a
rede de golpes termina de pé 16 de 18 martelos e 9 de 18 bênçãos; empurrado,
o corpo se afasta 0,5–0,9 m do centro do triângulo.

Esquiva lateral (rede própria), partindo da ginga: cinco testes de 15 minutos
e uma rodada de 4 horas (40 milhões de passos, taxa 1e-4). O tamanho chegou
à esquiva inteira em 20 minutos; a recompensa foi de 0,64 a 0,88 e a parte
dos episódios que chega ao fim do relógio de 0,52 a 0,94. No teste de 5
minutos (a ginga da melhor rede, esquiva inteira pedida a cada 5 s, cinco
andamentos de 60 a 130 bpm) terminou de pé 58 de 60 esquivas, com uma queda
só (a 60 bpm). A 66/90/120 bpm, com a esquiva a cada 5 s por 1 minuto em
cada um, 36 de 36 sem queda; a rodada anterior, treinada só até 60 % da
esquiva, caía 33 vezes no tamanho inteiro. Sozinha, a rede da esquiva ginga
como a da ginga (0 quedas de 66 a 130 bpm, pé a 2,4–3,0 cm do canto). Nas
imagens o corpo segue a referência com ~2° de erro nas juntas, desce ~0,06 s
atrás dela, o pé de trás dá o passo (2,5–7,6 cm do chão) em vez de arrastar,
a mão livre roça o chão ~0,1 s no fundo e a ginga depois volta limpa (raiz a
0,04–0,08 m da referência).

Golpes de novo (martelo e bênção), com a mesma receita da esquiva: a rede
dos golpes ginga 2,5 s depois do golpe, treina 3 s disso, 30 % dos episódios
começam na ginga e o triângulo vai para debaixo dos pés no fim do golpe. A
rede anterior não sabia gingar (sozinha, 35 quedas em 3 minutos, o pé a
~14 cm do canto), e as quedas vinham 1,4–2,5 s depois do golpe. Partindo da
ginga, 15 minutos não saíram do menor tamanho; partindo da rede anterior,
um teste de 15 minutos e uma rodada de 4 horas (taxa 1e-4): o tamanho chegou
ao golpe inteiro em 30 minutos e a recompensa foi de 0,52 a 0,74 (ainda
subindo no fim). No teste de 5 minutos (golpe inteiro a cada 5 s, cinco
andamentos de 60 a 130 bpm) termina de pé martelo 23 de 28 e bênção 24 de
27, com 13 quedas (a rede anterior, no mesmo código: 11 de 30, 21 de 29 e 38
quedas). Sozinha, a nova rede ginga sem cair, mas com o pé a 5–13 cm do
canto. Nas imagens o martelo e a bênção leem como os de verdade (~3° de
erro nas juntas) e voltam para uma ginga limpa; a bênção da perna direita
sobe ~15 cm a menos que a referência, e o pé de apoio gira menos que o da
referência no martelo e escorrega até 0,2 m.

Golpes com a postura das fotos (ver **Golpes** em [README-MOVES.md](README-MOVES.md): calcanhar de apoio no
alto no giro do martelo, tronco mais inclinado, bênção inclinada para trás, e
os termos `strike` e `pivot` na recompensa). Medida no pico de cada golpe, a
rede anterior não virava o quadril no martelo (1–16° contra 77° da
referência; o pé de apoio girava 6–10°, às vezes para o lado errado) e a
perna saía para o lado (65–89° da frente). Dois testes de 15 minutos (o
segundo já com `pivot`, que tirou o giro do pé do lado errado) e uma rodada
de 4 horas (taxa 1e-4; a recompensa foi de 0,63 a 0,79, `strike` 0,84 →
0,91, `pivot` 0,82 → 0,92). Com o `best.pt` o martelo vira: quadril 64° e
−50° (referência ±72°), pé de apoio 38° e −28° (27°), perna a 19° e 40° da
frente (18°), a 1,25–1,43 m; a bênção fica na referência (1,31–1,37 m,
tronco 15–23° para trás). Na CPU o equilíbrio ficava na beira da barra (martelo
11 de 16 e bênção 14 de 16 no `best.pt`, ou 15 de 17 e 10 de 17 no `last.pt`);
continuando no MJX em GPU L4 (512 ambientes, ~7 900 passos/s) até a amplitude
inteira (`mjx_kicks5`, 35 min a 1e-4) seguido de um polimento curto com menos
ruído (`mjx_kicks6`, 20 min a 3e-5, `log_std` −2,6), o `last.pt` termina de pé
martelo 15 de 16 (1 queda) e bênção 14 de 16 (1 queda) no teste de 3 minutos
(66, 90 e 120 bpm; barra: 14 de 16), e entrou no app.

