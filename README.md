## Movie Review Using LDAvis

![alt tag](https://github.com/kastureyS/BA-Assignments/blob/master/Hotel%20reviews.png)

### Preprocessing the data

Before fitting a topic model, we need to tokenize the text. We remove punctuation and some common stop words. In particular, we use the english stop words from the SMART information retrieval system, available in the R package tm.

```s
# read in some stopwords:
library(tm)
stop_words <- stopwords("SMART")
stop_words <- c(stop_words, "room", "hotel", "bed", "bathroom", "london", "us", "get", "got", "said", "the", "also", "just","for", "can", "may", "now", "year")
stop_words <- tolower(stop_words)

# pre-processing:
hotel_reviews <- gsub("'", "", hotel_reviews) # remove apostrophes
hotel_reviews <- gsub("[[:punct:]]", " ", hotel_reviews)  # replace punctuation with space
hotel_reviews <- gsub("[[:cntrl:]]", " ", hotel_reviews)  # replace control characters with space
hotel_reviews <- gsub("^[[:space:]]+", "", hotel_reviews) # remove whitespace at beginning of documents
hotel_reviews <- gsub("[[:space:]]+$", "", hotel_reviews) # remove whitespace at end of documents
hotel_reviews <- gsub("[^a-zA-Z -]", " ", hotel_reviews) # allows only letters
hotel_reviews <- tolower(hotel_reviews)  # force to lowercase
hotel_reviews <- hotel_reviews[hotel_reviews != ""] # remove blank docs

# tokenize on space and output as a list:
doc.list <- strsplit(hotel_reviews, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""] #table with names
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
   index <- match(x, vocab)
   index <- index[!is.na(index)]
   rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

```

### Using the R package 'lda' for model fitting
The object documents is a length-575 list where each element represents one document, according to the specifications of the lda package. After creating this list, we compute a few statistics about the corpus:

```s
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (575)
W <- length(vocab)  # number of terms in the vocab (193)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [11, 6, 4, 2 ...]
N <- sum(doc.length)  # total number of tokens in the data (2297)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [118,110,99,53,50,39 ...]

```

Next, we set up a topic model with 10 topics, relatively diffuse priors for the topic-term distributions (ηη = 0.02) and document-topic distributions (αα = 0.02), and we set the collapsed Gibbs sampler to run for 3,000 iterations . This block of code takes about 5.614015 secs to run.

```s
# MCMC and model tuning parameters:
K <- 10 #number of topics
G <- 3000 #number of iteration
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  #About bout 5.614015 secs

```

### Visualizing the fitted model with LDAvis

```s
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

```
We save these, along with ϕϕ, θθ, and vocab, in a list as the data object hotel_reviews_for_LDA, which is included in the LDAvis package.

```s
hotel_reviews_for_LDA <- list(phi = phi,
                    theta = theta,
                    doc.length = doc.length,
                    vocab = vocab,
                    term.frequency = term.frequency)
```

Now we're ready to call the createJSON() function in LDAvis. This function will return a character string representing a JSON object used to populate the visualization.

```s
library(LDAvis)
library(servr)
json <- createJSON(phi = hotel_reviews_for_LDA$phi, 
                  theta = hotel_reviews_for_LDA$theta, 
                  doc.length = hotel_reviews_for_LDA$doc.length, 
                  vocab = hotel_reviews_for_LDA$vocab, 
                  term.frequency = hotel_reviews_for_LDA$term.frequency)
```
The serVis() function can take json and serve the result in a variety of ways. Here we'll write json to a file within the 'vis' directory 

(http://127.0.0.1:4321/#topic=0&lambda=1&term=)

```s
serVis(json, out.dir = 'vis', open.browser = TRUE)
```

