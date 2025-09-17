// Vanilla JavaScript Conversational Search Application
// Using Typesense's built-in Conversational Search (RAG) capabilities

// Global conversation state
let conversationHistory = [];
let currentConversationId = null;
let conversationModelId = 'conv-model-1'; // This will need to be created in Typesense

// Typesense configuration for federated search
const typesenseInstantsearchAdapter = new TypesenseInstantSearchAdapter({
    server: {
        apiKey: "xyz",
        nodes: [
            {
                host: "localhost",
                port: "8108", 
                protocol: "http",
            },
        ],
        cacheSearchResultsForSeconds: 2 * 60,
    },
    // Common search parameters for all collections
    additionalSearchParameters: {
        highlight_fields: "content,document_title",
        snippet_threshold: 30,
        highlight_affix_num_tokens: 4,
        per_page: 20,
    },
    // Collection-specific search parameters
    collectionSpecificSearchParameters: {
        unstructured_fixed_size: {
            query_by: "content,document_title",
        },
        unstructured_semantic: {
            query_by: "content,document_title",
        },
        unstructured_sliding_langchain: {
            query_by: "content,document_title",
        },
        unstructured_sliding_unstructured: {
            query_by: "content,document_title",
        },
    },
});

// Conversation state variables
let conversationEnabled = true;
let isCheckingConversation = false;

// Perform conversational search using Typesense RAG
async function performConversationalSearch(query) {
    if (!conversationEnabled) {
        return performStandardSearch(query);
    }
    
    try {
        const searchParams = {
            searches: [
                {
                    collection: "unstructured_fixed_size",
                    query_by: "content,document_title",
                    exclude_fields: "embedding"
                },
                {
                    collection: "unstructured_semantic", 
                    query_by: "content,document_title",
                    exclude_fields: "embedding"
                },
                {
                    collection: "unstructured_sliding_langchain",
                    query_by: "content,document_title", 
                    exclude_fields: "embedding"
                },
                {
                    collection: "unstructured_sliding_unstructured",
                    query_by: "content,document_title",
                    exclude_fields: "embedding"
                }
            ]
        };
        
        // Add conversation parameters to URL
        const url = new URL('http://localhost:8108/multi_search');
        url.searchParams.append('q', query);
        url.searchParams.append('conversation', 'true');
        url.searchParams.append('conversation_model_id', conversationModelId);
        
        if (currentConversationId) {
            url.searchParams.append('conversation_id', currentConversationId);
        }
        
        const response = await fetch(url.toString(), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-TYPESENSE-API-KEY': 'xyz'
            },
            body: JSON.stringify(searchParams)
        });
        
        const results = await response.json();
        
        // Handle conversation response
        if (results.conversation) {
            currentConversationId = results.conversation.conversation_id;
            
            // Add to conversation history
            addToConversationHistory(query, {
                answer: results.conversation.answer,
                totalResults: results.results.reduce((sum, r) => sum + r.found, 0),
                processingTime: results.results[0]?.search_time_ms || 0
            });
            
            // Display conversational answer
            displayConversationalAnswer(results.conversation.answer);
        }
        
        // Display regular search results across strategies
        displayFederatedResults(results.results);
        
        return results;
        
    } catch (error) {
        console.error('Conversational search error:', error);
        // Fallback to standard search
        return performStandardSearch(query);
    }
}

// Perform standard federated search (fallback)
function performStandardSearch(query) {
    // This triggers the existing InstantSearch.js federated search
    const searchInput = document.querySelector('#searchbox input');
    if (searchInput && searchInput.value !== query) {
        searchInput.value = query;
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
}

// Display conversational answer from LLM
function displayConversationalAnswer(answer) {
    const conversationContainer = document.getElementById('conversation-history');
    if (!conversationContainer) return;
    
    // Create or update the current answer display
    let answerElement = document.querySelector('.current-conversation-answer');
    if (!answerElement) {
        answerElement = document.createElement('div');
        answerElement.className = 'current-conversation-answer';
        conversationContainer.appendChild(answerElement);
    }
    
    answerElement.innerHTML = `
        <div class="llm-answer">
            <div class="answer-label">💬 AI Assistant</div>
            <div class="answer-text">${answer}</div>
        </div>
    `;
    
    // Scroll to show the answer
    answerElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update UI for conversational mode
function updateUIForConversationalMode() {
    const placeholder = document.querySelector('#searchbox input');
    if (placeholder) {
        placeholder.placeholder = "Ask a question and get an AI-powered answer with supporting evidence (e.g., 'What is dynamic backtracking?')";
    }
    
    const suggestions = document.querySelector('.input-suggestions');
    if (suggestions) {
        suggestions.innerHTML = `
            <span class="suggestion-label">🤖 Try asking (AI-powered):</span>
            <button class="suggestion-btn" onclick="askQuestion('What is dynamic backtracking and how does it work?')">What is dynamic backtracking and how does it work?</button>
            <button class="suggestion-btn" onclick="askQuestion('Explain the advantages of constraint satisfaction algorithms')">Explain the advantages of constraint satisfaction algorithms</button>
            <button class="suggestion-btn" onclick="askQuestion('How do search algorithms handle large solution spaces?')">How do search algorithms handle large solution spaces?</button>
        `;
    }
}

// Update UI for standard mode
function updateUIForStandardMode() {
    const placeholder = document.querySelector('#searchbox input');
    if (placeholder) {
        placeholder.placeholder = "Ask a question about the documents (e.g., 'What is dynamic backtracking?', 'How do search algorithms work?')";
    }
    
    const suggestions = document.querySelector('.input-suggestions');
    if (suggestions) {
        suggestions.innerHTML = `
            <span class="suggestion-label">Try asking:</span>
            <button class="suggestion-btn" onclick="askQuestion('What is dynamic backtracking?')">What is dynamic backtracking?</button>
            <button class="suggestion-btn" onclick="askQuestion('How do search algorithms work?')">How do search algorithms work?</button>
            <button class="suggestion-btn" onclick="askQuestion('Explain constraint satisfaction problems')">Explain constraint satisfaction problems</button>
        `;
    }
}

// Handle question asking from UI
function askQuestion(question) {
    const searchInput = document.querySelector('#searchbox input');
    if (searchInput) {
        searchInput.value = question;
        handleSearch(question);
    }
}

// Main search handler - routes to conversational or standard search
async function handleSearch(query) {
    if (!query.trim()) return;
    
    // Show loading state
    showSearchLoading();
    
    try {
        if (conversationEnabled) {
            await performConversationalSearch(query);
        } else {
            performStandardSearch(query);
        }
    } catch (error) {
        console.error('Search error:', error);
        hideSearchLoading();
    }
}

// Add conversation item to history
function addToConversationHistory(question, response) {
    conversationHistory.push({
        id: Date.now().toString(),
        question: question,
        answer: response.answer || '',
        timestamp: new Date().toISOString(),
        totalResults: response.totalResults || 0,
        processingTime: response.processingTime || 0
    });
    
    // Update conversation display
    updateConversationDisplay();
}

// Update the conversation history display
function updateConversationDisplay() {
    const conversationContainer = document.getElementById('conversation-history');
    if (!conversationContainer || conversationHistory.length === 0) return;
    
    const historyHtml = conversationHistory.map(item => `
        <div class="conversation-item">
            <div class="question-bubble">
                <strong>You:</strong> ${item.question}
            </div>
            ${item.answer ? `
                <div class="answer-bubble">
                    <strong>AI:</strong> ${item.answer}
                    <div class="conversation-meta">
                        ${item.totalResults} results found in ${item.processingTime}ms
                    </div>
                </div>
            ` : ''}
        </div>
    `).join('');
    
    conversationContainer.innerHTML = `
        <div class="conversation-header">
            <h3>Conversation History</h3>
            <button onclick="clearConversationHistory()" class="clear-btn">Clear</button>
        </div>
        ${historyHtml}
    `;
}

// Clear conversation history
function clearConversationHistory() {
    conversationHistory = [];
    currentConversationId = null;
    const conversationContainer = document.getElementById('conversation-history');
    if (conversationContainer) {
        conversationContainer.innerHTML = '';
    }
}

// Show/hide loading states
function showSearchLoading() {
    const statsContainer = document.getElementById('stats');
    if (statsContainer) {
        statsContainer.innerHTML = '<div class="search-loading">🔍 Searching and analyzing documents...</div>';
    }
}

function hideSearchLoading() {
    // Loading will be hidden when results are displayed
}

// Display federated search results (existing format)
function displayFederatedResults(results) {
    // This function maintains the existing display format for search results
    // The results array contains search results from each collection/strategy
    console.log('Federated search results:', results);
    
    // Let InstantSearch.js handle the display by updating its state
    // This preserves the existing UI while adding conversational features
}

// Initialize InstantSearch
const search = instantsearch({
    indexName: "unstructured_fixed_size", // Primary index
    searchClient: typesenseInstantsearchAdapter.searchClient,
});

// Add debugging for connection issues
search.on('error', (error) => {
    console.error('InstantSearch Error:', error);
    // Show user-friendly error message
    const statsContainer = document.getElementById('stats');
    if (statsContainer) {
        statsContainer.innerHTML = `
            <div style="color: red; padding: 20px; text-align: center; background: #fee; border: 1px solid #fcc; border-radius: 8px;">
                <strong>Connection Error:</strong> Cannot connect to search service. 
                <br>Please ensure Typesense is running on localhost:8108.
                <br><small>Error: ${error.message || 'Unknown error'}</small>
            </div>
        `;
    }
});

search.on('render', () => {
    console.log('Search rendered successfully');
});

// Legacy function - keeping for compatibility with existing calls
function askQuestion(question) {
    console.log('askQuestion called with:', question);
    // Set the search box value and trigger search
    const searchInput = document.querySelector('#searchbox input');
    if (searchInput) {
        console.log('Found search input, setting value to:', question);
        searchInput.value = question;
        // Use the new search handler
        handleSearch(question);
        searchInput.focus();
    } else {
        console.error('Could not find search input element');
    }
}

function processConversationalQuery(query) {
    currentQuery = query;
    
    // For now, we'll use the query as-is since Typesense handles natural language well
    // In the future, we could add query processing/expansion here
    return query;
}

// Group hits by document_id for document-level display
function groupHitsByDocument(hits) {
    const documentGroups = {};
    
    // Debug: log the first hit to see available fields
    if (hits.length > 0) {
        console.log('First hit structure:', hits[0]);
        console.log('Available keys:', Object.keys(hits[0]));
        // Check if there's a text_match score
        if (hits[0].text_match) {
            console.log('Text match score:', hits[0].text_match);
        }
        if (hits[0].text_match_info) {
            console.log('Text match info:', hits[0].text_match_info);
        }
    }
    
    hits.forEach(hit => {
        const docId = hit.document_id || 'unknown';
        if (!documentGroups[docId]) {
            documentGroups[docId] = {
                document_id: docId,
                document_title: hit.document_title || 'Untitled Document',
                chunks: [],
                totalScore: 0,
                relevanceScore: 0,
                allChunkCount: 0,
                tokenRange: { min: Infinity, max: 0 }
            };
        }
        documentGroups[docId].chunks.push(hit);
        
        // Accumulate scores for averaging - try different score fields
        let score = 0;
        if (hit.text_match) {
            score = parseFloat(hit.text_match);
        } else if (hit.text_match_info && hit.text_match_info.score) {
            score = parseFloat(hit.text_match_info.score);
        } else if (hit._rankingInfo && hit._rankingInfo.score) {
            score = parseFloat(hit._rankingInfo.score);
        } else if (hit.score) {
            score = parseFloat(hit.score);
        } else if (hit._score) {
            score = parseFloat(hit._score);
        }
        documentGroups[docId].totalScore += score;
        
        // Track token count range
        const tokenCount = hit.token_count || 0;
        if (tokenCount > 0) {
            documentGroups[docId].tokenRange.min = Math.min(documentGroups[docId].tokenRange.min, tokenCount);
            documentGroups[docId].tokenRange.max = Math.max(documentGroups[docId].tokenRange.max, tokenCount);
        }
        
        // Estimate total chunks in document (this is approximate based on chunk indices)
        const chunkIndex = hit.chunk_index || 0;
        documentGroups[docId].allChunkCount = Math.max(documentGroups[docId].allChunkCount, chunkIndex + 1);
    });
    
    // Calculate average relevance score and sort documents
    return Object.values(documentGroups)
        .map(doc => {
            doc.relevanceScore = doc.chunks.length > 0 ? doc.totalScore / doc.chunks.length : 0;
            
            // Sort chunks by relevance and keep top 3 most relevant for preview
            doc.chunks = doc.chunks
                .sort((a, b) => {
                    const getScore = (hit) => {
                        if (hit.text_match) return parseFloat(hit.text_match);
                        if (hit.text_match_info && hit.text_match_info.score) {
                            return parseFloat(hit.text_match_info.score);
                        }
                        if (hit._rankingInfo && hit._rankingInfo.score) {
                            return parseFloat(hit._rankingInfo.score);
                        }
                        if (hit.score) return parseFloat(hit.score);
                        if (hit._score) return parseFloat(hit._score);
                        return 0;
                    };
                    return getScore(b) - getScore(a);
                })
                .slice(0, 3);
                
            // Clean up token range if no valid tokens found
            if (doc.tokenRange.min === Infinity) {
                doc.tokenRange.min = 0;
            }
            
            return doc;
        })
        .sort((a, b) => b.relevanceScore - a.relevanceScore);
}

// Hit template function for rendering document-level results in conversational format
function renderDocumentHit(document, strategyName) {
    const getHighlightedTitle = (chunk) => {
        if (chunk._highlightResult && chunk._highlightResult.document_title && chunk._highlightResult.document_title.value) {
            return chunk._highlightResult.document_title.value;
        }
        return document.document_title;
    };

    const getHighlightedContent = (chunk) => {
        if (chunk._highlightResult && chunk._highlightResult.content && chunk._highlightResult.content.value) {
            return chunk._highlightResult.content.value;
        }
        return chunk.content || 'No content available';
    };

    // Calculate chunk statistics
    const tokenCounts = document.chunks.map(chunk => chunk.token_count || 0).filter(count => count > 0);
    const chunkIndices = document.chunks.map(chunk => chunk.chunk_index || 0);
    
    // Use the precomputed token range from grouping function
    const minTokens = document.tokenRange ? document.tokenRange.min : 
        (tokenCounts.length > 0 ? Math.min(...tokenCounts) : 0);
    const maxTokens = document.tokenRange ? document.tokenRange.max : 
        (tokenCounts.length > 0 ? Math.max(...tokenCounts) : 0);
    const totalChunks = document.allChunkCount || document.chunks.length;
    
    // Show the most relevant chunk content as a preview
    const topChunk = document.chunks[0];
    const previewContent = getHighlightedContent(topChunk);
    
    // Try different ways to get the score
    let topChunkScore = 0;
    if (topChunk.text_match) {
        topChunkScore = topChunk.text_match;
    } else if (topChunk.text_match_info && topChunk.text_match_info.score) {
        topChunkScore = topChunk.text_match_info.score;
    } else if (topChunk._rankingInfo && topChunk._rankingInfo.score) {
        topChunkScore = topChunk._rankingInfo.score;
    } else if (topChunk.score) {
        topChunkScore = topChunk.score;
    } else if (topChunk._score) {
        topChunkScore = topChunk._score;
    } else if (document.relevanceScore) {
        topChunkScore = document.relevanceScore;
    }
    
    const scoreDisplay = parseFloat(topChunkScore).toFixed(3);

    return `
        <div class="document-answer">
            <div class="answer-header">
                <div class="strategy-badge">${strategyName}</div>
                <div class="relevance-score">Relevance: ${scoreDisplay}</div>
            </div>
            <div class="document-title">${getHighlightedTitle(topChunk)}</div>
            <div class="answer-content">
                <div class="content-preview">${previewContent}</div>
            </div>
            <div class="answer-meta">
                <span class="chunk-info">${document.chunks.length} relevant section${document.chunks.length !== 1 ? 's' : ''}</span>
                <span class="token-info">${minTokens > 0 ? (minTokens !== maxTokens ? `${minTokens}-${maxTokens}` : minTokens) : 'N/A'} tokens</span>
                <span class="document-id">Doc: ${document.document_id}</span>
            </div>
        </div>
    `;
}

// Configure search widgets
search.addWidgets([
    // Conversational Search Box
    instantsearch.widgets.searchBox({
        container: "#searchbox",
        placeholder: "Ask a question about the documents (e.g., 'What is dynamic backtracking?', 'How do search algorithms work?')",
        showLoadingIndicator: true,
        searchAsYouType: false, // Wait for user to finish typing question
        showSubmit: true,
        showReset: true
    }),

    // Stats Display with conversational context
    instantsearch.widgets.stats({
        container: "#stats",
        templates: {
            text: ({ nbHits, processingTimeMS, query }) => {
                if (query) {
                    const processedQuery = processConversationalQuery(query);
                    // Add to conversation history when we have results
                    setTimeout(() => {
                        const resultsSummary = {
                            totalResults: nbHits,
                            processingTime: processingTimeMS,
                            query: query
                        };
                        addToConversationHistory(query, resultsSummary);
                    }, 100);
                    
                    return `<div class="stats-conversational">
                        <div class="current-question">📝 "${query}"</div>
                        <div class="results-stats">Found ${nbHits.toLocaleString()} relevant documents in ${processingTimeMS}ms</div>
                    </div>`;
                }
                return '<div class="stats-prompt">Ask a question to search through the documents</div>';
            }
        }
    }),

    // Fixed Size Strategy Results
    instantsearch.widgets.index({ indexName: "unstructured_fixed_size" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-fixed",
                templates: {
                    item: (hit) => {
                        if (hit._isDocumentGroup) {
                            return renderDocumentHit(hit._documentGroup, "Fixed Size");
                        }
                        return '';
                    },
                    empty: '<div class="empty-state">No relevant content found with this chunking strategy</div>',
                },
                transformItems(items) {
                    const documents = groupHitsByDocument(items);
                    // Return only one item per document to fix pagination
                    return documents.map(doc => ({
                        ...doc.chunks[0], // Use the top chunk as the base
                        _isDocumentGroup: true,
                        _documentGroup: doc,
                        objectID: `doc_${doc.document_id}` // Unique ID for the document
                    }));
                }
            }),
        ]),

    // Semantic Strategy Results  
    instantsearch.widgets.index({ indexName: "unstructured_semantic" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-semantic",
                templates: {
                    item: (hit) => {
                        if (hit._isDocumentGroup) {
                            return renderDocumentHit(hit._documentGroup, "Semantic");
                        }
                        return '';
                    },
                    empty: '<div class="empty-state">No relevant content found with this chunking strategy</div>',
                },
                transformItems(items) {
                    const documents = groupHitsByDocument(items);
                    return documents.map(doc => ({
                        ...doc.chunks[0],
                        _isDocumentGroup: true,
                        _documentGroup: doc,
                        objectID: `doc_${doc.document_id}`
                    }));
                }
            }),
        ]),

    // Sliding LangChain Strategy Results
    instantsearch.widgets.index({ indexName: "unstructured_sliding_langchain" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-sliding-lc",
                templates: {
                    item: (hit) => {
                        if (hit._isDocumentGroup) {
                            return renderDocumentHit(hit._documentGroup, "Sliding (LangChain)");
                        }
                        return '';
                    },
                    empty: '<div class="empty-state">No relevant content found with this chunking strategy</div>',
                },
                transformItems(items) {
                    const documents = groupHitsByDocument(items);
                    return documents.map(doc => ({
                        ...doc.chunks[0],
                        _isDocumentGroup: true,
                        _documentGroup: doc,
                        objectID: `doc_${doc.document_id}`
                    }));
                }
            }),
        ]),

        // Sliding Unstructured Strategy Results
    instantsearch.widgets.index({ indexName: "unstructured_sliding_unstructured" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-sliding-un",
                templates: {
                    item: (hit) => {
                        if (hit._isDocumentGroup) {
                            return renderDocumentHit(hit._documentGroup, "Sliding (Unstructured)");
                        }
                        return '';
                    },
                    empty: '<div class="empty-state">No relevant content found with this chunking strategy</div>',
                },
                transformItems(items) {
                    const documents = groupHitsByDocument(items);
                    return documents.map(doc => ({
                        ...doc.chunks[0],
                        _isDocumentGroup: true,
                        _documentGroup: doc,
                        objectID: `doc_${doc.document_id}`
                    }));
                }
            }),
        ]),

    // Pagination (disabled since we're showing documents, not chunks)
    // instantsearch.widgets.pagination({
    //     container: "#pagination",
    //     padding: 2,
    //     showFirst: false,
    //     showLast: false,
    // }),
]);

// Start the search
search.start();

// Initialize conversational search capabilities
document.addEventListener('DOMContentLoaded', function() {
    // Initialize conversational search
    initializeConversationalSearch();
    
    const searchBox = document.querySelector('#searchbox input');
    if (searchBox) {
        // Handle search input for conversational mode
        searchBox.addEventListener('input', function(e) {
            const query = e.target.value;
            if (query.length > 2) {
                // Debounce for live search
                clearTimeout(window.searchTimeout);
                window.searchTimeout = setTimeout(() => {
                    handleSearch(query);
                }, 300);
            }
            
            // Add loading indicators
            const containers = ['#hits-fixed', '#hits-semantic', '#hits-sliding-lc', '#hits-sliding-un'];
            containers.forEach(container => {
                const element = document.querySelector(container);
                if (element) {
                    element.classList.add('loading');
                    setTimeout(() => element.classList.remove('loading'), 300);
                }
            });
        });
        
        // Handle Enter key for immediate search
        searchBox.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                const query = e.target.value.trim();
                if (query) {
                    handleSearch(query);
                }
            }
        });
    }
    
    // Make functions globally available
    window.askQuestion = askQuestion;
    window.clearConversationHistory = clearConversationHistory;
    window.handleSearch = handleSearch;
});

console.log('Conversational Document Search Application Loaded - Ask questions to find relevant content across chunking strategies!');
