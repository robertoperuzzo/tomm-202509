// Vanilla JavaScript Federated Search Application
// Using InstantSearch.js with Typesense adapter

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
        per_page: 6, // Limit results per collection for better UX
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

// Initialize InstantSearch
const search = instantsearch({
    indexName: "unstructured_fixed_size", // Primary index
    searchClient: typesenseInstantsearchAdapter.searchClient,
});

// Hit template function for rendering search results
function renderHit(hit, strategyName) {
    // Handle highlighting - check if _highlightResult exists
    const getHighlightedContent = () => {
        if (hit._highlightResult && hit._highlightResult.content && hit._highlightResult.content.value) {
            return hit._highlightResult.content.value;
        }
        // Fallback to original content if highlighting isn't available
        return hit.content || 'No content available';
    };

    const getHighlightedTitle = () => {
        if (hit._highlightResult && hit._highlightResult.document_title && hit._highlightResult.document_title.value) {
            return hit._highlightResult.document_title.value;
        }
        return hit.document_title || 'Untitled Document';
    };

    return `
        <div class="hit">
            <div class="hit-strategy">${strategyName}</div>
            <div class="hit-meta">Chunk ${hit.chunk_index || 'N/A'} • ${hit.token_count || 'N/A'} tokens</div>
            <div class="hit-content">${getHighlightedContent()}</div>
            <div class="hit-source">${getHighlightedTitle()}</div>
        </div>
    `;
}

// Configure search widgets
search.addWidgets([
    // Search Box
    instantsearch.widgets.searchBox({
        container: "#searchbox",
        placeholder: "Search across chunking strategies...",
        showLoadingIndicator: true,
    }),

    // Stats Display
    instantsearch.widgets.stats({
        container: "#stats",
        templates: {
            text: ({ nbHits, processingTimeMS, query }) => {
                if (query) {
                    return `${nbHits.toLocaleString()} results found in ${processingTimeMS}ms`;
                }
                return 'Enter a search term to compare chunking strategies';
            }
        }
    }),

    // Fixed Size Strategy Results
    instantsearch.widgets.index({ indexName: "unstructured_fixed_size" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-fixed",
                templates: {
                    item: (hit) => renderHit(hit, "Fixed Size"),
                    empty: '<div class="empty-state">No fixed size results</div>',
                },
            }),
        ]),

    // Semantic Strategy Results  
    instantsearch.widgets.index({ indexName: "unstructured_semantic" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-semantic",
                templates: {
                    item: (hit) => renderHit(hit, "Semantic"),
                    empty: '<div class="empty-state">No semantic results</div>',
                },
            }),
        ]),

    // Sliding LangChain Strategy Results
    instantsearch.widgets.index({ indexName: "unstructured_sliding_langchain" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-sliding-lc",
                templates: {
                    item: (hit) => renderHit(hit, "Sliding (LangChain)"),
                    empty: '<div class="empty-state">No sliding LangChain results</div>',
                },
            }),
        ]),

    // Sliding Unstructured Strategy Results
    instantsearch.widgets.index({ indexName: "unstructured_sliding_unstructured" })
        .addWidgets([
            instantsearch.widgets.hits({
                container: "#hits-sliding-un",
                templates: {
                    item: (hit) => renderHit(hit, "Sliding (Unstructured)"),
                    empty: '<div class="empty-state">No sliding unstructured results</div>',
                },
            }),
        ]),

    // Pagination (using primary index)
    instantsearch.widgets.pagination({
        container: "#pagination",
        padding: 2,
        showFirst: false,
        showLast: false,
    }),
]);

// Start the search
search.start();

// Add some loading feedback
document.addEventListener('DOMContentLoaded', function() {
    const searchBox = document.querySelector('#searchbox input');
    if (searchBox) {
        searchBox.addEventListener('input', function() {
            // Add a subtle loading indicator when typing
            const containers = ['#hits-fixed', '#hits-semantic', '#hits-sliding-lc', '#hits-sliding-un'];
            containers.forEach(container => {
                const element = document.querySelector(container);
                if (element) {
                    element.classList.add('loading');
                    setTimeout(() => element.classList.remove('loading'), 300);
                }
            });
        });
    }
});

console.log('Federated Search Application Loaded - Ready to search across chunking strategies!');
