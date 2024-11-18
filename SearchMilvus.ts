import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const milvusClient = new MilvusClient({
  address: '127.0.0.1:19530',
});

const COLLECTION_NAME = 'text_embeddings';

async function getEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  return response.data[0].embedding;
}

async function main() {
  try {
    await milvusClient.useDatabase({
      db_name: "my_database",
    });

    console.log("Loading collection...");
    await milvusClient.loadCollection({
      collection_name: COLLECTION_NAME,
    });

    const searchText = "cdk";
    const searchVector = await getEmbedding(searchText);

    console.log("Searching similar texts...");
    const searchResponse = await milvusClient.search({
      collection_name: COLLECTION_NAME,
      vector: searchVector,
      search_params: {
        anns_field: "vector",
        topk: 5,
        metric_type: "COSINE",
        params: JSON.stringify({ nprobe: 128 }),
      },
      output_fields: ["text"],
    });

    console.log("Search results:");
    searchResponse.results.forEach((result, i) => {
      console.log(`${i + 1}. Text: ${result.text}`);
      console.log(`   Similarity: ${result.score}\n`);
    });

  } catch (error) {
    console.error("Error in search process:", error);
  } finally {
    await milvusClient.releaseCollection({
      collection_name: COLLECTION_NAME,
    });
    await milvusClient.closeConnection();
  }
}

main();
