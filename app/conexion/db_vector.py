from pinecone import Pinecone
from typing import List, Dict, Optional, Union

class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        # Inicializa el cliente de Pinecone con la API key
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = None
        
        # Verifica si el índice ya existe
        if not self.pc.has_index(self.index_name):
            print("creando indice")
            self.create_index()
        else:
            print("indice existente")
            self.index = self.pc.Index(self.index_name)
    
    def create_index(self):
        """Crea un índice denso con incrustación integrada"""
        self.pc.create_index_for_model(
            name=self.index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
        print(f"Índice '{self.index_name}' creado con éxito.")
    
    def prepare_records(self, input_records: list):
        """
        Transforma registros con metadata anidado a estructura aplanada
        
        Args:
            input_records: Lista de diccionarios con estructura:
                {
                    "id": str,
                    "chunk_text": str,
                    "metadata": dict
                }
        
        Returns:
            Lista de diccionarios con estructura aplanada:
                {
                    "_id": str,
                    "chunk_text": str,
                    ... (todos los campos del metadata original)
                }
        """
        output_records = []
        
        for record in input_records:
            # Crear nuevo registro con campos base
            transformed = {
                "_id": record["id"],
                "chunk_text": record["chunk_text"]
            }
            
            # Aplanar los metadatos (si existen)
            if "metadata" in record and isinstance(record["metadata"], dict):
                for key, value in record["metadata"].items():
                    transformed[key] = value
            
            output_records.append(transformed)
        
        return output_records
    

    def upsert_records(self, namespace: str, records: list):
        """Inserta o actualiza registros en un namespace específico"""
        
        self.index.upsert_records(namespace, records)
        print(f"{len(records)} registros insertados en '{namespace}'.")

    def check_index_stats(self):
        """Obtiene estadísticas del índice"""
        return self.index.describe_index_stats()

    # ------------------------- MÉTODOS DE BÚSQUEDA -------------------------

    def semantic_search(self, query: str, top_k: int, namespace: str, 
                       filter: Optional[Dict] = None, fields: Optional[List[str]] = None):
        """
        Búsqueda semántica (densa) usando embeddings
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados a devolver
            namespace: Namespace donde buscar
            filter: Diccionario con filtros de metadatos
            fields: Campos a incluir en los resultados
            
        Returns:
            Resultados de la búsqueda semántica
        """
        params = {
            "namespace": namespace,
            "query": {
                "top_k": top_k,
                "inputs": {'text': query}
            }
        }
        
        if filter:
            params["query"]["filter"] = filter
        if fields:
            params["fields"] = fields
            
        return self.index.search(**params)

    def lexical_search(self, query: str, top_k: int, namespace: str, 
                      filter: Optional[Dict] = None, fields: Optional[List[str]] = None):
        """
        Búsqueda léxica (sparse) basada en términos exactos
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados a devolver
            namespace: Namespace donde buscar
            filter: Diccionario con filtros de metadatos
            fields: Campos a incluir en los resultados
            
        Returns:
            Resultados de la búsqueda léxica
        """
        params = {
            "namespace": namespace,
            "query": {
                "top_k": top_k,
                "inputs": {'text': query},
                "sparse": True  # Indica que es búsqueda léxica
            }
        }
        
        if filter:
            params["query"]["filter"] = filter
        if fields:
            params["fields"] = fields
            
        return self.index.search(**params)

    def hybrid_search(self, query: str, top_k: int, namespace: str, 
                     filter: Optional[Dict] = None, fields: Optional[List[str]] = None,
                     rerank_model: str = "bge-reranker-v2-m3"):
        """
        Búsqueda híbrida (semántica + léxica) con reranking
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados a devolver
            namespace: Namespace donde buscar
            filter: Diccionario con filtros de metadatos
            fields: Campos a incluir en los resultados
            rerank_model: Modelo a usar para reranking
            
        Returns:
            Resultados de la búsqueda híbrida con reranking
        """
        params = {
            "namespace": namespace,
            "query": {
                "top_k": top_k,
                "inputs": {'text': query},
                "hybrid": True  # Activa búsqueda híbrida
            },
            "rerank": {
                "model": rerank_model,
                "top_n": top_k,
                "rank_fields": fields or ["chunk_text"]
            }
        }
        
        if filter:
            params["query"]["filter"] = filter
        if fields:
            params["fields"] = fields
            
        return self.index.search(**params)

    def search_by_metadata(self,query_, filter: Dict, top_k: int, namespace: str,
                          fields: Optional[List[str]] = None):
        """
        Búsqueda exclusivamente por metadatos (sin query vectorial)
        
        Args:
            filter: Diccionario con filtros de metadatos
            top_k: Número de resultados a devolver
            namespace: Namespace donde buscar
            fields: Campos a incluir en los resultados
            
        Returns:
            Resultados que coinciden con los filtros de metadatos
        """
        # Usamos una query vacía con solo filtros
        return self.index.search(
            namespace=namespace,

            query={
                "inputs": {"text":query_},
                "top_k": top_k,
                "filter": filter
            },
            fields=fields
        )

    # ------------------------- MÉTODOS AVANZADOS -------------------------

    def search_with_custom_embedding(self, embedding: List[float], top_k: int, 
                                    namespace: str, filter: Optional[Dict] = None):
        """
        Búsqueda usando un embedding vectorial personalizado
        
        Args:
            embedding: Vector de embedding precalculado
            top_k: Número de resultados a devolver
            namespace: Namespace donde buscar
            filter: Diccionario con filtros de metadatos
            
        Returns:
            Resultados más similares al embedding proporcionado
        """
        params = {
            "namespace": namespace,
            "query": {
                "top_k": top_k,
                "vector": embedding
            }
        }
        
        if filter:
            params["query"]["filter"] = filter
            
        return self.index.search(**params)

    def delete_by_metadata(self, namespace: str, filter: Dict):
        """
        Elimina registros basados en filtros de metadatos
        
        Args:
            namespace: Namespace donde eliminar
            filter: Diccionario con filtros de metadatos
        """
        self.index.delete(namespace=namespace, filter=filter)
        print(f"Registros eliminados en '{namespace}' con filtro: {filter}")



