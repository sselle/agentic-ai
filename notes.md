
memory = gibt verschiedene Arten von Speicher (MemorySaver == in-memory). Für richtige Deployments sollte der Memory auf eine DB zeigen (Redis)

+------------------+----------------+------------------+
|                  | MemorySaver    | SqliteSaver      |
+------------------+----------------+------------------+
| Speicherort      | RAM            | Festplatte       |
| Überlebt Neustart| Nein           | Ja               |
| Setup            | keine Abh.     | SQLite           |
| Wann sinnvoll    | Entwicklung    | Production       |
+------------------+----------------+------------------+

thread-id / memory: informationen im Speicher werden immer per Thread abgespeichert > neue Thread-ID heißt, dass der Agent keinen Zugriff mehr auf vorherige Konversationen hat 

tools - avoid too many tools. It can cause context overload. Define just the right amount requried for the task of the Agent. Dynamic tool selection can prevent this. However, it increases the complexity of the implementation 