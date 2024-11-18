import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const milvusClient = new MilvusClient({
  address: '127.0.0.1:19530',
});

const COLLECTION_NAME = 'text_embeddings';
const EMBEDDING_DIM = 1536;

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

    const hasCollection = await milvusClient.hasCollection({
      collection_name: COLLECTION_NAME,
    });

    // if (hasCollection.value) {
    //   console.log("Dropping existing collection...");
    //   await milvusClient.dropCollection({
    //     collection_name: COLLECTION_NAME,
    //   });
    //   await new Promise(resolve => setTimeout(resolve, 2000));
    // }

    console.log("Creating collection...");
    await milvusClient.createCollection({
      collection_name: COLLECTION_NAME,
      fields: [
        {
          name: "id",
          data_type: DataType.Int64,
          is_primary_key: true,
        },
        { 
          name: "text", 
          data_type: DataType.VarChar, 
          max_length: 1024
        },
        { 
          name: "vector", 
          data_type: DataType.FloatVector, 
          dim: EMBEDDING_DIM
        }
      ],
    });
    await new Promise(resolve => setTimeout(resolve, 2000));

    console.log("Creating index...");
    await milvusClient.createIndex({
      collection_name: COLLECTION_NAME,
      field_name: 'vector',
      extra_params: {
        index_type: 'IVF_FLAT',
        metric_type: 'COSINE',
        params: JSON.stringify({ nlist: 128 })
      }
    });
    await new Promise(resolve => setTimeout(resolve, 3000));

    console.log("Loading collection...");
    await milvusClient.loadCollection({
      collection_name: COLLECTION_NAME,
    });
    await new Promise(resolve => setTimeout(resolve, 2000));

    const text = "At the core of the city’s operation lies a network of advanced AI systems that oversee every aspect of its infrastructure—from transportation and energy to public services and security. These AI systems are meticulously designed to ensure the city functions seamlessly, efficiently, and sustainably, showcasing the potential of AI in creating a harmonious urban environment.";
    const embedding = await getEmbedding(text);

    console.log("Inserting data...");
    const insertResponse = await milvusClient.insert({
      collection_name: COLLECTION_NAME,
      fields_data: [{
        id: 7,
        text: text,
        vector: embedding
      }]
    });
    console.log("Insert response:", JSON.stringify(insertResponse, null, 2));

    console.log("Flushing data...");
    await milvusClient.flush({
      collection_names: [COLLECTION_NAME],
    });

  } catch (error) {
    console.error("Error in main process:", error);
  } finally {
    await milvusClient.releaseCollection({
      collection_name: COLLECTION_NAME,
    });
    await milvusClient.closeConnection();
  }
}

main();
