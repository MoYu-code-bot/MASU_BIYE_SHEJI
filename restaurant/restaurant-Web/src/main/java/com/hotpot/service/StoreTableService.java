package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.StoreTable;

import java.util.List;

public interface StoreTableService extends IService<StoreTable> {

    List<StoreTable> listByStoreId(Long storeId);
}
