# Movimentos do pipeline

Continuação do [pipeline](README.md#pipeline): os golpes, a esquiva, o cair e levantar e a escolha de perna e momento.

- **Golpes (kicks.py, env_choreo.py).** O martelo e a bênção partem dos
  quadros-chave do app, que foram ajustados olhando as duas câmeras do vídeo
  nos pontos-chave de cada golpe, e conferidos com descrições técnicas: no
  martelo o joelho sobe como na bênção, o pé de apoio gira 90–180° na ponta,
  o quadril fica alinhado com a perna, o corpo inclina um pouco para trás e a
  canela estala no fim, acertando com o peito do pé na altura do ombro/cabeça
  ([Lalaue — martelo](https://lalaue.com/moves/martelo/)); a bênção é um
  empurrão frontal com a sola ou o calcanhar, na altura do peito, saindo da
  base ([Lalaue — bênção](https://lalaue.com/moves/bencao)). Ajuste feito por
  isso: no golpe do martelo o giro vai a 90° e a perna sobe mais (o peito do
  pé chega a ~1,2 m). Depois, comparados com fotos dos dois golpes
  ([martelo, Abadá](https://commons.wikimedia.org/wiki/File:CapoeiraMartelo_ST_05.jpg),
  [martelo num batizado](https://commons.wikimedia.org/wiki/File:Batizado_Planeta_Capoeira_Martelo.jpg),
  [bênção, Abadá](https://commons.wikimedia.org/wiki/File:CapoeiraBencao_ST_05.jpg)):
  o martelo girava o pé de apoio 75° com ele chapado no chão — o corpo
  simulado só gira um pé chapado arrastando os quatro cantos, e por isso não
  girava (o quadril virava 1–16° em vez de 77°, e a perna saía para o lado);
  agora o calcanhar de apoio sobe ~7 cm e o giro é na ponta do pé, o tronco
  inclina 27° para longe da perna (antes 18°) e o peito do pé chega a 1,37 m.
  A bênção inclinava o tronco 5° para a frente; agora inclina 15° para trás,
  com o pé de apoio chapado, e a sola chega à altura do peito (ponta do pé a
  1,36 m, antes 1,13 m). Agora o martelo vem de um capoeirista de verdade
  num vídeo (Capoeira Nantes, `ZCLVQIfaPHE`, um martelo de cada perna): o
  GVHMR (`hmr_modal.py`, na GPU do Modal) tira o corpo em 3D com o caminho
  no chão, o `retarget.py` ajusta o esqueleto e o `plant_clip.py` planta os
  pés onde o vídeo pisa, com o peso sobre o apoio e o giro na ponta do pé.
  O trecho usado é o replay em câmera lenta do vídeo (4 vezes mais devagar;
  o `clip_speed.py` o devolve ao tempo real, com 4 vezes os quadros da tomada
  normal), cortado de logo antes de o joelho subir até o corpo pousar e
  assentar de pé (1,4 s na direita, 2,0 s na esquerda, reamostrados para um
  comprimento só): o corte anterior parava com o joelho recolhido no alto e
  voltava para a base por mistura, e o corpo caía 0,3–1,2 s depois do golpe.
  O pouso é uma base larga e baixa com o tronco inclinado para a frente, como
  no vídeo (conferido sobre a imagem do GVHMR). A esquerda é o espelho da
  direita (`clip_mirror.py`): o ajuste do vídeo inteiro perde o capoeirista
  no fim do replay, e o ajuste só do replay pousa a esquerda com o ponto de
  momento zero a 39 cm do apoio (19 cm na direita) — a rede ficava de pé nela
  3 vezes menos. A mistura
  com a base acontece antes e depois do trecho do vídeo, não por cima dele. Cada golpe é amostrado a 30 quadros/s para as duas
  pernas (`scripts/sample-clips.ts`), convertido em ângulos do modelo e
  plantado por IK: o pé de apoio fica no seu canto da frente o tempo todo;
  quando gira (no martelo, na queixada, na meia-lua de compasso), gira na
  ponta: o calcanhar sobe conforme o pé vira, de 15° (ainda chapado) a 35°
  (7 cm), a ponta fica parada e o peso vai para cima dela. O corpo
  simulado não gira um pé chapado, e no martelo do vídeo, com o calcanhar a
  1–4 cm do chão, o quadril não virava: a rede caía em 34 de 35 martelos. O pé que chuta sai do canto de trás e volta para ele num arco de até 8 cm
  do chão, indo de onde a pose o leva para o canto aos poucos (antes, o pé
  no ar era puxado direto para o canto: o chão teria de puxar o corpo para
  baixo, e a ponta do pé arrastava até 1,2 m no chão).
  A ginga entra no golpe a partir da base com a perna que chuta atrás (fase 0
  para a direita, 0,5 para a esquerda, com até ±0,1 de folga), congela a fase
  enquanto o golpe acontece, mistura a pose da ginga com a do golpe nos
  primeiros e nos últimos 0,35 s e segue a ginga dali. O golpe se ancora no
  pé em que o corpo está de fato apoiado (até 25 cm do canto): a ginga nunca
  pisa exatamente no canto, e um golpe equilibrado sobre o canto com o pé a
  10 cm dali derrubava o corpo. Quem espera o momento é a rede da ginga; a
  rede dos golpes assume quando o golpe começa e devolve 2,5 s depois (como a
  da esquiva, abaixo; devolvendo em 1 s, ~60 % das quedas vinham 1,4–2,5 s
  depois do golpe, já na rede da ginga), e quando o golpe acaba o triângulo
  vai para debaixo dos pés. O treino dos golpes começa no golpe: metade dos
  episódios no primeiro quadro (com a pose um pouco perturbada e o pé um
  pouco fora do canto, como na passagem da ginga), metade no meio dele —
  ou, em 30 %, na ginga esperando o momento — e cada um segue 3 s de ginga
  depois. Os tamanhos sobem (40 → 60 → 80 → 100 % do golpe) quando 75 % dos
  golpes inteiros terminam de pé. A política recebe qual golpe, qual perna, o
  quanto do golpe já passou e a mistura.
- **Esquiva lateral (esquiva.py, env_choreo.py).** Também sai da base e volta para
  ela, como um golpe, e tem rede própria (`--skill esquiva`): da base com a
  perna direita atrás, o corpo desce de uma vez sobre a perna da frente
  dobrada enquanto a de trás abre esticada para o lado num passo baixo
  (2–2,5 cm do chão), o tronco deita sobre o joelho com a cabeça para fora,
  para a esquerda, os olhos no camarada e o antebraço direito sobre a
  cabeça; segura, sobe na perna da frente e o pé de trás volta para o canto.
  Desce tão rápido quanto a gravidade e as pernas deixam (o quadril cai com
  no máximo 7 m/s² e freia com no máximo 9 m/s²): a cabeça passa de 1 m em
  ~0,33 s (um golpe chega em 0,3–0,4 s) e o fundo vem em ~0,5 s, quadril a
  ~0,5 m e cabeça a ~0,76 m, pés a ~1 m um do outro — nos cantos da frente do
  triângulo (0,68 m) uma perna esticada seguraria o quadril a 0,8 m. Sobe
  mais devagar (no máximo 4 m/s² para cima e 6 m/s² para frear: subir de uma
  perna só bem dobrada é o difícil, e um corpo que não acompanha a subida
  senta), e o pé de trás só volta para o canto com o quadril quase em cima;
  1,8 s ao todo. Parte dos quadros-chave do app (`esquiva-lateral.yaml`), que a IK
  planta: o pé da frente fica no seu canto, o de trás sai do canto num passo
  sem tranco e pousa chapado; o centro de massa recebe um empurrão curto do
  pé de trás e rola como um pêndulo até o pé da frente enquanto o de trás
  está no ar (o ponto de momento zero fica no pé da frente); altura do
  quadril e da cabeça, a inclinação do tronco, solas chapadas e pernas e
  mãos longe do chão também entram no ajuste. Tamanhos para o currículo:
  40–100 % da esquiva. A política recebe as mesmas entradas de um golpe e
  mais uma, no fim, que diz que é a esquiva. A rede da esquiva continua
  gingando 2,5 s depois dela antes de devolver o corpo à rede da ginga, e treina 3 s de ginga depois de cada esquiva: devolvido 1 s
  depois, o corpo ainda não estava numa ginga que a rede da ginga
  segurasse, e caía 1,5–2,7 s depois da esquiva. Treinada só a partir da
  esquiva, a rede esquecia a ginga (sozinha, caía 4–12 vezes em 30 s a cada
  toque; a da ginga, nenhuma): por isso 30 % dos episódios começam na ginga
  e esperam o momento de entrar, e o treino parte da rede da ginga com
  taxa de aprendizado menor (1e-4). E quando a esquiva acaba o
  triângulo da ginga vai para debaixo dos pés, onde eles de fato estão (como
  depois de levantar do chão; sem girar — girar pela linha entre os pés
  virava a ginga 8–20° a cada esquiva, e o corpo saía andando atrás): o passo largo nunca
  volta exato para o canto, e uma ginga que continua pisando num triângulo
  a 0,1–0,4 m dali sai andando e cai. Fora de uma queda, quadril
  abaixo de 0,4 m ou cabeça abaixo de 0,8 m é queda — ou, onde a referência
  desce mais que isso, 0,3 m abaixo dela.
- **Cair e levantar (floor.py, falls.py, env_choreo.py).** Capoeirista que perde o
  equilíbrio não desaba: caindo para trás, senta na queda de quatro (mãos e
  pés no chão, barriga para cima); para o lado, desce na negativa (uma perna
  dobrada por baixo, a outra esticada, a mão do lado no chão); para a frente,
  apara a queda nas mãos e cai de joelhos. As poses de pouso das duas
  primeiras vêm dos quadros-chave do app; a terceira e os três caminhos de
  volta têm os quadros-chave no próprio `floor.py`, feitos para que o corpo
  nunca precise se jogar do chão para cima: mão ou pé só sai do chão com o
  peso já sobre o que fica. Da queda de quatro, as mãos empurram o quadril
  para a frente, por cima dos pés, até a cocorinha, e o corpo sobe dali; da
  negativa, a perna esticada se recolhe arrastando o calcanhar e planta o pé
  à frente, a mão empurra o peso para ele e o corpo sobe de uma base baixa
  com a perna dobrada atrás; de quatro apoios, um pé vem à frente ao lado da
  mão, o quadril passa por cima dele, as mãos empurram o chão numa base baixa
  e o corpo sobe com o outro pé atrás. A IK põe pouso e caminho no chão de
  trás para a frente: o último quadro é a base da ginga no
  triângulo, e em cada quadro anterior as mãos, joelhos e pés que sustentam o
  peso tocam o chão sem escorregar (só o calcanhar arrastado desliza) — um apoio
  que sai do chão fica onde tocou no pouso, e o peso passa para os outros aos
  poucos —, com o centro de massa sobre eles, e já sobre os que ficam antes
  que um deles saia. Onde os quadros-chave sobem rápido demais para um corpo de
  torque limitado, o clipe se alonga: o centro de massa nunca sobe a mais de
  0,8 m/s nem anda a mais de 1 m/s (o caminho de volta dura 3,5–6,2 s). O
  `falls.py` grava 3000 perdas de equilíbrio de verdade da ginga e dos golpes
  treinados, com empurrões de 120–450 N de um lado qualquer. Perder o
  equilíbrio é algo além de mãos e pés no chão, a cabeça baixa, ou o quadril
  inclinado 0,65 rad além do da referência (só a inclinação: o giro do
  martelo sobre o pé de apoio não é queda), 25 cm afundado ou 0,5 m longe
  dela. Para onde o corpo vai é para onde o centro de massa passa do meio dos
  pés, mais a velocidade do quadril, visto de onde o corpo está virado (não a
  inclinação do quadril, que na ginga já pende para a frente): para a frente,
  mãos e joelhos; para trás, queda de quatro; para o lado, negativa, dobrando
  a perna que carrega o peso (o pé mais perto de para onde o corpo tomba) e
  esticando a livre — dobrar a livre estica a de apoio, e o corpo só pode
  girar por cima dela —, mas um tronco já dobrado mais de 0,9 rad para a frente (do quadril
  à cabeça) vai para mãos e joelhos de qualquer jeito: sentar numa negativa
  com a cabeça já sobre as mãos é mergulhar de cabeça. A referência vai da
  pose do momento até a pose de pouso em 0,5 s (rápido no começo e
  assentando no pouso, mas nunca à frente da gravidade: o quadril não desce
  mais do que desceria em queda livre a partir da velocidade que tinha, e
  a pose sobe o que for preciso para nenhum pé, joelho ou mão passar do chão
  no caminho; os
  braços vão logo buscar o chão: caindo para trás, as mãos estão atrás do
  quadril e perto do chão em 0,3 s),
  virada para onde o corpo está virado e um pouco adiante na direção da queda;
  ao pousar, o caminho de volta começa onde o corpo de fato está, sobe e
  entrega a ginga, que continua onde o corpo levantou (o triângulo vai junto).
  No treino, só a cabeça e o pescoço não podem tocar o chão (o episódio
  acaba se, depois de pousar, o corpo se afasta 0,6 m do caminho de volta, se
  fica 1 s deitado enquanto o caminho de volta espera por ele, ou se, descendo ou
  logo depois de pousar, a cabeça fica 30 cm abaixo da da referência — o
  corpo que segura o quadril no alto e tomba por cima dele mergulha de
  cabeça e só bate um instante depois), e durante a queda
  parte da nota vem de descer devagar (quadril, peito e cabeça sem despencar)
  e com a cabeça acima do quadril (o quadril desce primeiro; mergulhar de
  cabeça é como ela bate no chão). As quedas gravadas começam devagar: de
  início com o corpo parado na pose em que perdeu o equilíbrio, depois com
  35 %, 70 % e toda a velocidade, a cada vez que metade delas termina de pé.
  Os episódios começam numa queda gravada (50 %), na pose de pouso (20 %) ou em
  qualquer ponto do caminho de volta (30 %, em qualquer lugar e direção do
  chão) e seguem 2 s de ginga depois de levantar. A rede de recuperação é a
  terceira rede: assume ao perder o equilíbrio e devolve 1 s depois de
  levantar. Por enquanto ela quase nunca levanta (0 de 29 quedas no teste, ver o
  [resultado do treino](README-TRAINING.md#resultado-do-treino)), e o app põe
  o corpo de pé de novo depois de 1,2 s.
- **Perna e momento (choose.py).** Depois do treino, para cada golpe, cinco
  andamentos e 16 momentos do pedido dentro do ciclo, a política tenta, em
  simulação, as duas pernas e cinco momentos de entrada a partir de uma ginga
  já estabilizada. Fica a opção que melhor equilibra ficar de pé (até 1 s
  depois do golpe), acompanhar bem o golpe e responder rápido (perde 0,15 de
  nota por segundo de espera). O app só consulta essa tabela.
