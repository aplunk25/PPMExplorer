# PPMExplorer

PPMExplorer is a project aimed to drawing new insights from images and captions in the *Pompei: Pitture e Mosaic* (PPM) encyclopedia set. The project uses [Svelte](https://svelte.dev/) for building the web application, [Microsoft Azure AI Vision API](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/) for generating image and caption feature vectors, and [Microsoft Azure Cosmos DB](https://azure.microsoft.com/en-us/services/cosmos-db/) for storing and indexing the data, as well as [OpenAI CLIP](https://openai.com/index/clip/) for Captions and Images. 

## Background

Dr. Cindy Roulette, a former Ph.D. student at the University of Arkansas, developed a similar application using Unity in 2020. Our objective for this project is to recreate the application in a web-based format and modernize its features to incorporate advancements in computer vision, natural language processing, and artificial intelligence.

### What are feature vectors?

Feature vectors are numerical representations of data that can be used to compare and analyze the similarity between different data points. In the context of this project, we query vectorized user input against a database of vectorized images and captions to find the most similar matches.

![Feature Vector Similarity Example](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.couchbase.com%2Fblog%2Fwp-content%2Fuploads%2F2024%2F02%2Fimage2-2-1024x794.png&f=1&nofb=1&ipt=5286deea902048596b4720e481b805794ea5a15129ea6e2efd1b51d8ec17b00c&ipo=images)
[Source](https://www.couchbase.com/blog/what-is-vector-search/)

## Local Development

To set up the project locally, follow these steps:

1. **Install Node.js:**

  If you don't have Node.js installed, download and install the latest version from the [official website](https://nodejs.org/).

2. **Clone the repository:**

  ```bash
  git clone https://github.com/cadedupont/ppmexplorer.git
  cd ppmexplorer
  ```

3. **Install dependencies:**

  ```bash
  npm install
  ```

4. **Obtain necessary API keys:**

  [Contact the project owners](#contact) to obtain the required API keys and add them to a `.env` file in the root directory of the project.

5. **Run the development server:**

  ```bash
  npm run dev
  ```

  or

  ```bash
  npm run dev -- --open
  ```

  to open the application in your default browser.

## Local CLIP sample search

The home page searches the bundled `350Package/embeddings/random_samples` dataset with
OpenCLIP ViT-B/32 (OpenAI weights). Install the Python dependencies before starting the web app:

```bash
python -m pip install -r 350Package/requirements.txt
npm run dev
```

The first search loads the model and may download its weights. Set `PYTHON_EXECUTABLE` if your
Python installation is not available as `python` (for example, to the path of your virtual
environment's Python executable). The two similarity sliders set minimum cosine similarity;
an image-caption pair is returned when either its image or its caption meets its threshold.

This sample does not include location geometry, so search results are not plotted on the map.

## Contact

For any inquiries, please contact the project owners:

| Name | Email |
| --- | --- |
| Dr. David Fredrick | [dfredric@uark.edu](mailto:dfredric@uark.edu) |
| Dr. Cindy Roulette | [cindy.e.roulette@gmail.com](mailto:cindy.e.roulette@gmail.com) |
| Alex Plunk | [acplunk@uark.edu](mailto:acplunk@uark.edu) |
| Brodie Baugh | [bbaugh@uark.edu](mailto:bbaugh@uark.edu) |
| Logan Davis | [logand@uark.edu](mailto:logand@uark.edu) |
| Josh Godoy | [jgodoy@uark.edu](mailto:jgodoy@uark.edu) |
