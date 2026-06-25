package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.TableType;

import java.util.List;

public interface TableTypeService extends IService<TableType> {

    List<TableType> listByStoreId(Long storeId);
}
