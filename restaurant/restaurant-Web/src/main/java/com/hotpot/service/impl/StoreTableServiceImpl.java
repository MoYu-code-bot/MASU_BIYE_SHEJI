package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.entity.StoreTable;
import com.hotpot.mapper.StoreTableMapper;
import com.hotpot.service.StoreTableService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class StoreTableServiceImpl extends ServiceImpl<StoreTableMapper, StoreTable> implements StoreTableService {

    @Override
    public List<StoreTable> listByStoreId(Long storeId) {
        return list(new LambdaQueryWrapper<StoreTable>()
                .eq(StoreTable::getStoreId, storeId)
                .eq(StoreTable::getStatus, 1));
    }
}
