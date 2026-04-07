# **1.	Definición del Problema: Leyes de Protección Ambiental en Parques Nacionales de Argentina**
## a) Qué queremos resolver (Caso de Uso):
Nuestro objetivo principal es crear una herramienta que ayude a entender y consultar fácilmente las leyes y normativas sobre protección ambiental y conservación de la naturaleza que se aplican en los Parques Nacionales de Argentina. Este sistema usará Inteligencia Artificial para que cualquier persona que necesite información legal específica de estas áreas protegidas no tenga que perder tiempo buscando entre muchísimos documentos. Queremos simplificar el acceso a todo ese conocimiento que hoy está disperso en diferentes leyes, decretos y resoluciones.
## b) ¿Para quién es esta herramienta? (Usuario Objetivo):
Nuestro sistema RAG está pensado para quienes necesitan información legal ambiental precisa y relevante, como:
	Estudiantes y profesores de temas ambientales, derecho o biología: Para investigar, hacer trabajos o analizar casos.
	Investigadores y técnicos (tanto de gobierno como privados): Les servirá para armar informes, proyectos de conservación o estudios de impacto ambiental dentro de los Parques Nacionales.
	Personal de Parques Nacionales (guardaparques, funcionarios, etc.): Para consultar rápidamente las normas ante situaciones del día a día (por ejemplo, saber qué está permitido, qué está prohibido, qué sanciones hay o cómo actuar en ciertos casos).
	Abogados especializados en derecho ambiental: Para encontrar más rápido los artículos o leyes que les interesan.
	Miembros de ONGs ambientalistas: Para comprender mejor el marco legal que protege nuestra biodiversidad.
        Público en general que desea consultar las leyes de Protección ambiental en Parques Nacionales de Argentinas,.
## c) ¿Qué tipo de preguntas podrá responder el sistema?:
La herramienta debe ser capaz de sacar información exacta de los documentos legales y, si es necesario, juntar datos de varios puntos de la normativa para dar una respuesta completa. No esperamos que dé opiniones o consejos legales complejos, sino que siempre remita a la ley o el artículo correspondiente.
Algunos ejemplos de preguntas que podría responder son:
	Sobre definiciones y conceptos:
## ¿Qué Ley General del Ambiente es mencionada como marco en materia de presupuestos mínimos para la Evaluación de Impacto Ambiental?
	Sobre lo que está prohibido o restringido:
	Sobre permisos y autorizaciones:
	Sobre multas y cómo proceder:
	Sobre comparar o ver el alcance de algo:
## ¿Por qué usar RAG para este problema? (Justificación):
Elegimos usar la arquitectura RAG porque la información legal es:
	Muy extensa y complicada: Buscar algo específico a mano en leyes y decretos es muy tedioso y fácil de equivocarse.
	Específica y técnica: El lenguaje legal es particular y las leyes se relacionan entre sí, algo que RAG puede manejar mejor.
	Cambia con el tiempo: Aunque las leyes principales no cambien seguido, sí aparecen decretos o resoluciones nuevas. Una búsqueda simple por palabras clave no sería suficiente, y un modelo de IA común podría inventar información al no estar actualizado.
	Necesitamos saber de dónde viene la respuesta: Es fundamental poder verificar la fuente (qué ley o artículo) de la información que nos dé el sistema.
# **2. Construcción del Corpus**



