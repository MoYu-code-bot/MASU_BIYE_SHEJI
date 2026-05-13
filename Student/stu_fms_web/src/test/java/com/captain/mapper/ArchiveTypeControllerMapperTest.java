package com.captain.mapper;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.vo.AuditVo;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class ArchiveTypeMapperTest {

    @Autowired
    private ArchiveTypeMapper archiveTypeMapper;

    @Test
    void testList(){

        IPage<AuditVo> list = archiveTypeMapper.list(new Page(2, 1));
        System.out.println("总页数"+list.getPages());
        System.out.println("总数据数"+list.getTotal());
        System.out.println("当前页"+list.getCurrent());
        System.out.println("数据"+list.getRecords());
    }

}