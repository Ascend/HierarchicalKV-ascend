# HierarchicalKV-ascend HashTableOptions

*说明：*

*若参数"是否支持"为是，"限制与说明"为"-"，说明此参数和原生参数支持度保持一致。*

| 参数名称                   | 是否支持 | 限制与说明     |
| -------------------------- | -------- | -------------- |
| init_capacity              | 是       | -              |
| max_capacity               | 是       | -              |
| max_hbm_for_vectors        | 是       | -              |
| max_bucket_size            | 是       | 必须大于等于16 |
| dim                        | 是       | -              |
| max_load_factor            | 是       | -              |
| block_size                 | 否       | -              |
| io_block_size              | 否       | -              |
| device_id                  | 是       | -              |
| io_by_cpu                  | 是       | -              |
| use_constant_memory        | 是       | -              |
| reserved_key_start_bit     | 否       | -              |
| num_of_buckets_per_alloc   | 是       | -              |
| api_lock                   | 是       | 默认值为false  |
| device_memory_pool         | 是       | -              |
| host_memory_pool           | 是       | -              |

# HierarchicalKV-ascend HashTable

*说明：*

*若API"是否支持"为是，"限制与说明"为"-"，说明此API和原生API支持度保持一致。*

| API名称            | 是否支持 | 限制与说明                                                      |
| ------------------ | -------- | --------------------------------------------------------------- |
| init               | 是       | -                                                               |
| insert_or_assign   | 是       | -                                                               |
| insert_and_evict   | 是       | -                                                               |
| accum_or_assign    | 是       | 不支持double、uint16数据类型                                    |
| find_or_insert     | 是       | -                                                               |
| find_or_insert*    | 是       | -                                                               |
| lock_keys          | 是       | -                                                               |
| unlock_keys        | 是       | -                                                               |
| assign             | 是       | -                                                               |
| assign_scores      | 是       | -                                                               |
| assign_values      | 是       | -                                                               |
| find               | 是       | -                                                               |
| find(value)        | 是       | -                                                               |
| find*              | 是       | -                                                               |
| find_and_update    | 是       | -                                                               |
| contains           | 是       | -                                                               |
| erase              | 是       | -                                                               |
| erase_if           | 是       | -                                                               |
| erase_if_v2        | 是       | 使用HOST DDR存储value时，不支持自定义函数对value进行内存操作    |
| clear              | 是       | -                                                               |
| export_batch       | 是       | -                                                               |
| export_batch_if    | 是       | -                                                               |
| export_batch_if_v2 | 是       | 使用HOST DDR存储value时，不支持自定义函数对value进行内存操作    |
| for_each           | 是       | 使用HOST DDR存储value时，不支持自定义函数对value进行内存操作    |
| empty              | 是       | -                                                               |
| size               | 是       | -                                                               |
| size_if            | 是       | -                                                               |
| capacity           | 是       | -                                                               |
| reserve            | 是       | -                                                               |
| load_factor        | 是       | -                                                               |
| set_max_capacity   | 是       | -                                                               |
| dim                | 是       | -                                                               |
| max_bucket_size    | 是       | -                                                               |
| bucket_count       | 是       | -                                                               |
| save               | 是       | -                                                               |
| load               | 是       | -                                                               |
| set_global_epoch   | 是       | -                                                               |
