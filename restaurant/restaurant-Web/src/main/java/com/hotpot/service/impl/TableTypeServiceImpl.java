package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.entity.TableType;
import com.hotpot.mapper.TableTypeMapper;
import com.hotpot.service.TableTypeService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TableTypeServiceImpl extends ServiceImpl<TableTypeMapper, TableType> implements TableTypeService {

    @Override
    public List<TableType> listByStoreId(Long storeId) {
        return list(new LambdaQueryWrapper<TableType>()
                .eq(TableType::getStoreId, storeId)
                .or()
                .isNull(TableType::getStoreId));
    }
}
