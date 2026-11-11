#include <stdint.h>

void greedy_match(
    const int64_t *order,
    const int64_t *edge_source,
    const int64_t *edge_target,
    int64_t edge_count,
    uint8_t *source_used,
    uint8_t *target_used,
    int64_t *match_by_source)
{
    for (int64_t position = 0; position < edge_count; ++position) {
        const int64_t edge = order[position];
        const int64_t source = edge_source[edge];
        const int64_t target = edge_target[edge];
        if (!source_used[source] && !target_used[target]) {
            source_used[source] = 1;
            target_used[target] = 1;
            match_by_source[source] = target;
        }
    }
}
