Machine learning
---

Při používání neuronových sítí je důležité spoustu aspektů
U vytváření modelu: vybrat vhodnou velikost modelu, vhodné vrstvy, parametry, jejich rozložení a propojení
  S tím jako funkcí (loss funkce) má neuronová síť konvergovat k výsledku a jakými metrikami se bude hodnotit kvalita
U trénování modelu: vhodné data, velikost dat, a správné rozložení učení
  S optimizérem mít epochy, kroky za epochu, validaci, a velikost seskupení dat při učení (batch)


Každá tato část ovlivňuje negativně model, pokud je moc malá, nebo i moc velká

velikost modelu - záleží, 120 MB model potřebuje přes 12 GB paměti pro učení
                          300 MB model potřebuje přes 65 GB paměti pro učení
                          1 GB model potřebuje přes 115 GB paměti pro učení




trénování
epoch ~5-20, málo epoch = model se neoptimalizuje, hodně epoch = zbytečné trenování 
kroky ~50-100, málo kroků = nedostatečný trénink, hodně kroků = model se přeučí na trénovacích datech
batch ~4-32, málý batch =  trénování bude pomalejší, ale zabírat méně paměti při trénování, velký batch = rechlejší trénovaní, ale větší pamětoví zatížení
