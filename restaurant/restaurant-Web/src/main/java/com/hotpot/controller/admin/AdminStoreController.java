package com.hotpot.controller.admin;

import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Store;
import com.hotpot.service.StoreService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
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
    public Result<PageResult<Store>> page(PageQuery query) {
        return Result.success(storeService.pageQuery(query));
    }

    @ApiOperation("新增门店")
    @PostMapping
    public Result<?> add(@RequestBody Store store) {
        storeService.save(store);
        return Result.success();
    }

    @ApiOperation("修改门店")
    @ApiImplicitParam(name = "storeId", value = "门店ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/update")
    public Result<?> update(@RequestParam Long storeId, @RequestBody Store store) {
        store.setId(storeId);
        storeService.updateById(store);
        return Result.success();
    }

    @ApiOperation("删除门店")
    @ApiImplicitParam(name = "storeId", value = "门店ID", required = true, dataType = "long", paramType = "query")
    @DeleteMapping("/delete")
    public Result<?> delete(@RequestParam Long storeId) {
        storeService.removeById(storeId);
        return Result.success();
    }
}
