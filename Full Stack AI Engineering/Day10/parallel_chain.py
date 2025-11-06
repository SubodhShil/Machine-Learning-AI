import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


from dotenv import load_dotenv
load_dotenv()


model = init_chat_model("groq:openai/gpt-oss-120b")


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text\n{text}',
    input_variables=['text'],
)


prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following texts\n{text}',
    input_variables=['text'],
)


prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document\nnotes -> {notes} and {quiz}',
    input_variables=['text', 'quiz'],
)

parser = StrOutputParser()


start_time = time.monotonic()

# Create the parallel chain
parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

merge_chain = prompt3 | model | parser
chain = parallel_chain | merge_chain


text = """
This article is about the religions that originated in the Indian subcontinent. For religious demographics of the Republic of India, see Religion in India. For the book, see The Religion of India. For the religions of indigenous peoples of North America, see Native American religions.
Indian religions, sometimes also termed Indic religions or Dharmic religions, are the religions that originated in the Indian subcontinent. These religions, which include Buddhism, Hinduism, Jainism, and Sikhism,[web 1][note 1] are also classified as Eastern religions. Although Indian religions are connected through the history of India, they constitute a wide range of religious communities, and are not confined to the Indian subcontinent.[web 1]

More information Religion, Population ...

Thumb
Symbols of major Indian religions
Indian religions as a percentage of world population
Hinduism (16.0%)
Buddhism (7.10%)
Sikhism (0.35%)
Jainism (0.06%)
Non-Indian religions and irreligion (76.5%)
Evidence attesting to prehistoric religion in the Indian subcontinent derives from scattered Mesolithic rock paintings. The Harappan people of the Indus Valley Civilisation, which lasted from 3300 to 1300 BCE (mature period 2600–1900 BCE), had an early urbanised culture which predates the Vedic religion.[5][better source needed]

The documented history of Indian religions begins with the historical Vedic religion, the religious practices of the early Indo-Aryan peoples, which were collected and later redacted into the Vedas, as well as the Agamas of Dravidian origin. The period of the composition, redaction, and commentary of these texts is known as the Vedic period, which lasted from roughly 1750 to 500 BCE.[6] The philosophical portions of the Vedas were summarised in Upanishads, which are commonly referred to as Vedānta, variously interpreted to mean either the "last chapters, parts of the Veda" or "the object, the highest purpose of the Veda".[7] The early Upanishads all predate the Common Era, five[note 2] of the eleven principal Upanishads were composed in all likelihood before the 6th century BCE,[8][9] and contain the earliest mentions of yoga and moksha.[10]

The śramaṇa period between 800 and 200 BCE marks a "turning point between the Vedic Hinduism and Puranic Hinduism".[11] The Shramana movement, an ancient Indian religious movement parallel to but separate from Vedic tradition, often defied many of the Vedic and Upanishadic concepts of soul (Atman) and the ultimate reality (Brahman). In the 6th century BCE, the Shramnic movement matured into Jainism[12] and Buddhism[13] and was responsible for the schism of Indian religions into two main philosophical branches of astika, which venerates Veda (e.g., six orthodox schools of Hinduism) and nastika (e.g., Buddhism, Jainism, Charvaka, etc.). However, both branches shared the related concepts of yoga, saṃsāra (the cycle of birth and death) and moksha (liberation from that cycle).[note 3][note 4][note 5]

The Puranic Period (200 BCE – 500 CE) and early medieval period (500–1100 CE) gave rise to new configurations of Hinduism, especially bhakti and Shaivism, Shaktism, Vaishnavism, Smarta, and smaller groups like the conservative Shrauta.

The early Islamic period (1100–1500 CE) also gave rise to new movements. Sikhism was founded in the 15th century on the teachings of Guru Nanak and the nine successive Sikh Gurus in Northern India.[web 2] The vast majority of its adherents originate in the Punjab region. During the period of British rule in India, a reinterpretation and synthesis of Hinduism arose, which aided the Indian independence movement.


Remove ads
History
See also: Outline of South Asian history, History of India, History of Hinduism, and History of Buddhism
Periodisation
Main article: Periodisation of Hinduism
Scottish historian James Mill, in his seminal work The History of British India (1817), distinguished three phases in the history of India, namely the Hindu, Muslim, and British periods. This periodisation has been criticised, for the misconceptions it has given rise to. Another periodisation is the division into "ancient, classical, medieval, and modern periods", although this periodisation has also received criticism.[16]

Romila Thapar notes that the division of Hindu-Muslim-British periods of Indian history gives too much weight to "ruling dynasties and foreign invasions",[17] neglecting the social-economic history which often showed a strong continuity.[17] The division in Ancient-Medieval-Modern overlooks the fact that the Muslim conquests took place between the eight and the fourteenth centuries, while the south was never completely conquered.[17] According to Thapar, a periodisation could also be based on "significant social and economic changes", which are not strictly related to a change of ruling powers.[18][note 6]

Smart and Michaels seem to follow Mill's periodisation, while Flood and Muesse follow the "ancient, classical, mediaeval and modern periods" periodisation. An elaborate periodisation may be as follows:[19]

Indian pre-history including Indus Valley Civilisation (until c. 1750 BCE)
Iron Age including Vedic period (c. 1750–600 BCE)
"Second Urbanisation" (c. 600–200 BCE)
Classical period (c. 200 BCE – 1200 CE)[note 7]
Pre-Classical period (c. 200 BCE – 320 CE)
"Golden Age" (Gupta Empire) (c. 320–650 CE)
Late-Classical period (c. 650–1200 CE)
Medieval period (c. 1200–1500 CE)
Early Modern (c. 1500–1850)
Modern period (British Raj and independence) (from c. 1850)
"""

result = chain.invoke({'text': text})
end_time = time.monotonic()
print(result)
chain.get_graph().print_ascii()
print(f"Time taken: {end_time - start_time} seconds")
