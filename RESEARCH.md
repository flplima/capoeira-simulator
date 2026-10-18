# Pesquisa: 20 movimentos básicos da capoeira

Não existe uma lista canônica de "20 movimentos". Cada grupo e cada linhagem tem
o seu currículo, e os nomes variam entre a Capoeira Angola, a Capoeira Regional
e a Contemporânea. Esta lista é o que há em comum entre:

- **As sequências de ensino de Mestre Bimba** — o primeiro método formal de
  ensino da capoeira, montado a partir de 17 a 19 movimentos básicos: ginga,
  meia-lua de frente, meia-lua de compasso, armada, aú, rolê, cocorinha,
  negativa, queixada, bênção, cabeçada, martelo, galopante, esquiva, joelhada,
  entre outros. O exame de admissão de Bimba era cocorinha, queda de rins e
  ponte.
- **Os currículos de iniciantes dos grupos atuais**, como resumidos por sites de
  ensino (Dende Arts, Lalaue, grupos universitários): primeiro a ginga e as
  esquivas, depois o jogo de chão (negativa, rolê, aú), depois os golpes
  circulares, os golpes diretos e a rasteira.
- **O glossário de técnicas da Wikipédia**, usado para a biomecânica de cada
  movimento.

Ficaram de fora os movimentos que exigem uma habilidade anterior ou cuidado de
verdade com o risco (macaco, queda de rins, s-dobrado, tesoura, aú sem mão,
parafuso). A ponte está no exame de admissão de Bimba, mas é mais um exercício
de preparo físico do que um movimento de jogo, então a vaga ficou com o
galopante (também de Bimba).

## A lista

A ordem é a mesma do app: fundamentos, depois defesas, depois o jogo de chão,
depois os ataques — dos circulares aos diretos e aos de curta distância —,
terminando na rasteira. Os quadros-chave são sempre escritos para o lado
direito; o app espelha para o esquerdo.

| # | Movimento | Categoria | O que é |
|---|---|---|---|
| 1 | **Ginga** | fundamento | O balanço constante dos pés — paralela, um pé atrás, paralela, o outro pé atrás — com o braço do lado da perna de trás protegendo o rosto. Todo movimento começa e termina nela. |
| 2 | **Esquiva lateral** | esquiva | Esquiva para o lado: pés afastados e paralelos, o corpo desce para um lado por baixo do golpe, um antebraço sobre a cabeça, a mão livre pode tocar o chão. |
| 3 | **Esquiva baixa** | esquiva | Esquiva baixa de frente: uma ginga exagerada e bem baixa — perna de trás ainda mais para trás, quadril baixo, peito sobre a coxa da frente, mão no chão, o outro antebraço protegendo o rosto. |
| 4 | **Cocorinha** | esquiva | Esquiva de cócoras: pés chapados, joelhos no peito, uma mão cobrindo a cabeça e a outra no chão. A primeira defesa que a maioria dos grupos ensina. |
| 5 | **Negativa** | esquiva | Esquiva baixa para um lado: a perna de apoio dobrada embaixo do corpo, a outra esticada, uma mão no chão e o braço livre protegendo o rosto. A porta de entrada do jogo de chão. |
| 6 | **Queda de quatro** | esquiva | Cair para trás apoiado em mãos e pés (posição de caranguejo) para escapar de um golpe alto, e voltar para a ginga. |
| 7 | **Rolê** | movimentação | Um giro baixo rente ao chão, mãos no chão, olhando o camarada por entre as pernas enquanto as costas viram; liga negativas e muda de posição na roda. |
| 8 | **Aú** | movimentação | A estrela da capoeira, que sai de uma esquiva, fechada e atenta; as pernas passam por cima do corpo e descem uma de cada vez. |
| 9 | **Meia-lua de frente** | ataque circular | A perna de trás, esticada, passa na frente do corpo de fora para dentro, na altura da cabeça. |
| 10 | **Queixada** | ataque circular | O inverso da meia-lua de frente — um passo à frente e a perna esticada passa de dentro para fora, batendo com a borda externa do pé. |
| 11 | **Armada** | ataque circular | Chute giratório: cruza o passo, vira as costas para o camarada e solta a perna esticada no giro do quadril, batendo com a borda externa do pé. |
| 12 | **Meia-lua de compasso** | ataque circular | Também *rabo de arraia*. Mãos no chão, olhos no camarada, a perna de trás varre um grande arco com o calcanhar; a força vem da mão batendo no chão e do giro do quadril. |
| 13 | **Bênção** | ataque direto | Também chapa de frente: um chute de empurrar, reto, com a sola ou o calcanhar no peito ou na barriga. |
| 14 | **Martelo** | ataque direto | Chute semicircular com o peito do pé ou a canela, em geral na cabeça, girando o pé de apoio. |
| 15 | **Chapa** | ataque direto | Chute lateral de empurrar (pisão): joelho sobe, o pé de apoio gira para longe, a sola empurra para o lado. |
| 16 | **Ponteira** | ataque direto | Chute frontal estalado com a planta do pé, dedos puxados para trás, tronco inclinando para trás. |
| 17 | **Joelhada** | ataque de proximidade | Golpe de joelho, para cima e para a frente, de perto; na quinta sequência de Bimba é dado com os braços protegendo a cabeça. |
| 18 | **Cabeçada** | ataque de proximidade | Golpe com a testa, normalmente numa barriga exposta — depois de um aú ou de uma esquiva. Raramente dada de verdade no jogo atual, mas é um fundamento. |
| 19 | **Galopante** | ataque de proximidade | O tapa de mão aberta e em concha de Bimba, no lado da cabeça (na orelha). |
| 20 | **Rasteira** | desequilibrante | A rasteira clássica: descer, mão no chão, e enganchar o calcanhar de apoio do camarada com o peito do pé, usando o próprio movimento dele. |

## Como os movimentos viraram dados

O app hoje mostra só os movimentos que o corpo simulado aprendeu (ginga,
martelo, bênção, esquiva lateral, e as quedas na queda de quatro e na
negativa); dos 20, ficaram os arquivos que servem de referência para o treino.
Os outros estão no histórico do git.

Cada movimento era um arquivo YAML em
[`public/data/movements/`](public/data/movements/) com os textos acima (resumo,
passos, dicas, fontes) e uma lista de quadros-chave — posições do corpo em
ângulos de articulação, em graus, cada uma num tempo em segundos. As técnicas
foram posadas num manequim de 1,75 m e conferidas de frente, de lado e em 3/4
numa folha de contato (`/sheet.html?m=<id>`). Os tempos miram o ritmo da
Regional (São Bento Grande): golpes de cerca de 1,5 a 2 s da ginga de volta à
ginga, com o golpe rápido e a preparação e a volta mais lentas.

As poses são uma ilustração para aprender as formas, não substituem um
professor: onde as fontes divergem (qual mão apoia a negativa, se o aú vai para
a frente ou para o lado), os arquivos seguem a variante mais comum da Regional
e dizem isso nas dicas.

## Fontes

- [List of capoeira techniques — Wikipedia](https://en.wikipedia.org/wiki/List_of_capoeira_techniques) — biomecânica de todos os movimentos acima.
- [The 8 Training Sequences of Mestre Bimba — Papoeira](https://papoeira.com/en/the-8-training-sequences-of-mestre-bimba/) — as sequências de Bimba e os movimentos básicos de que são feitas.
- [An Intro to Mestre Bimba, Part 2: The Foundation of Capoeira Regional — Papoeira](https://papoeira.com/en/an-intro-to-mestre-bimba-part-2-the-foundation-of-capoeira-regional/)
- [Capoeira Regional — Wikipedia](https://en.wikipedia.org/wiki/Capoeira_Regional)
- [Capoeira Regional — Portal São Francisco](https://www.portalsaofrancisco.com.br/esportes/capoeira-regional/amp) — o exame de admissão de Bimba (cocorinha, queda de rins, ponte).
- [10 basic capoeira movements for beginners — Dende Arts](https://dendearts.com/blog/10-basic-capoeira-movements-for-beginners/)
- [List of fundamental capoeira movements — Dende Arts](https://dendearts.com/blog/list-of-fundamental-capoeira-movements/)
- [Capoeira basics — Lalaue](https://www.lalaue.com/learn-capoeira/capoeira-basic)
- [Capoeira moves — University of Bristol Capoeira Society](https://bristol.ac.uk/depts/union/capoeira/moves.html)
- [Rasteira](https://en.wikipedia.org/wiki/Rasteira), [Bênção](https://en.wikipedia.org/wiki/B%C3%AAn%C3%A7%C3%A3o) e [Cabeçada](https://en.wikipedia.org/wiki/Cabe%C3%A7ada) — Wikipedia.
