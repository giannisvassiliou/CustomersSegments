# CustomersSegments
Timely detection of customer segment evolution is vital for organizations operating in dynamic markets. This work presents a production-ready framework that integrates online clustering with lifecycle event detection and real-time notification. The framework captures a range of segment dynamics—including births, deaths, merges, user drifts, and size changes—while providing confidence scores to support interpretability. In contrast to prior prototypes, the proposed system explicitly addresses deployment challenges such as thread safety, scalability, and workflow integration, thereby ensuring robustness in medium-velocity streaming environments. We evaluate the approach using a realistic synthetic e-commerce data streams, demonstrating responsiveness, stability, and practical relevance.We also utilize two algorithms: online Kmeans and Clustream and compare to periodically executed standard k-means. The architecture remains agnostic to the choice of clustering algorithm, facilitating extensibility. Overall, the framework transforms raw behavioral streams into actionable, low-latency insights suitable for real-world application esensuring reliable operation in medium to high-throughput environments processing up to 300 users per second (18,000 users per minute) in commodity hardware.


## CODE - Python:

thirteenthD - online kmeans
thirteenthD_BATCH_WARMSTART - batch (standard) kmeans
thirteenthD_clustream - CluStream

