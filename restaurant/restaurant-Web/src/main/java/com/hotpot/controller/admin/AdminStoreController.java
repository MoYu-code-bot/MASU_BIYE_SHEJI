package com.hotpot.controller.admin;

import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Store;
import com.hotpot.service.StoreService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Api(tags = "B端-门店管理")
@RestController
@RequestMapping("/admin/stores")
@RequiredArgsConstructor
public class AdminStoreController {

    private final StoreService storeService;

    @ApiOperation("分页查询门店")
    @GetMapping
    public Result<PageResult<Store>> page(@ApiParam("分页参数") PageQuery query) {
        return Result.success(storeService.pageQuery(query));
    }

    @ApiOperation("新增门店")
    @PostMapping
    public Result<?> add(@ApiParam("门店信息") @RequestBody Store store) {
        storeService.save(store);
        return Result.success();
    }

    @ApiOperation("修改门店")
    @PutMapping("/update")
    public Result<?> update(@ApiParam("门店ID") @RequestParam Long storeId,
                            @ApiParam("门店信息") @RequestBody Store store) {
        store.setId(storeId);
        storeService.updateById(store);
        return Result.success();
    }

    @ApiOperation("删除门店")
    @DeleteMapping("/delete")
    public Result<?> delete(@ApiParam("门店ID") @RequestParam Long storeId) {
        storeService.removeById(storeId);
        return Result.success();
    }
}
