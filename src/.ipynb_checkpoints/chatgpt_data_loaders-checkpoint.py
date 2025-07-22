import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import threading
import fsspec
import interp_utils

_thread_local = threading.local()

class llc4320_dataset(Dataset):
    def __init__(self, data_dir, mid_timestep, N_t, patch_coords,
                 infields, outfields,
                 in_mask_list, out_mask_list,
                 in_transform_list, out_transform_list,
                 standards=None, N=128, L_x=512e3, L_y=512e3,
                 squeeze=False, return_metadata=False,
                 return_masks=False, time_loading=False,
                 regrid_SWOT=False, cloud_rho=0.7):

        self.data_dir = data_dir
        self.mid_timestep = mid_timestep
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.infields = infields
        self.outfields = outfields
        self.in_mask_list = in_mask_list
        self.out_mask_list = out_mask_list
        self.in_transform_list = in_transform_list
        self.out_transform_list = out_transform_list
        self.N = N; self.L_x = L_x; self.L_y = L_y
        self.squeeze = squeeze
        self.return_meta_data = return_metadata
        self.return_masks = return_masks
        self.time_loading = time_loading
        self.regrid_SWOT = regrid_SWOT
        self.cloud_rho = cloud_rho

        if standards is None:
            standards = {"mean_ssh":0., "std_ssh":1., "mean_sst":0., "std_sst":1.}
        self.transforms = {
            "std_ssh_norm":   self._make_standardize(std=standards["std_ssh"]),
            "std_sst_norm":   self._make_standardize(std=standards["std_sst"]),
            "std_mean_ssh_norm": self._make_std_samplewise(std=standards["std_ssh"]),
            "std_mean_sst_norm": self._make_std_samplewise(std=standards["std_sst"]),
            "std_global_mean_ssh_norm": self._make_standardize(mean=standards["mean_ssh"], std=standards["std_ssh"]),
            "std_global_mean_sst_norm": self._make_standardize(mean=standards["mean_sst"], std=standards["std_sst"]),
            "no_transform": lambda x: x
        }

    def __len__(self):
        return self.patch_coords.shape[0]

    def __getitem__(self, idx):
        pid = str(int(self.patch_coords[idx,2])).zfill(3)
        coords = self.patch_coords[idx]
        if getattr(_thread_local, 'initialized', False) is False:
            self._init_worker_local()
        invar, inmask = self._load_fields(pid, self.infields,self.in_transform_list, self.in_mask_list)
        if self.outfields:
            outvar, outmask = self._load_fields(pid, self.outfields,self.out_transform_list, self.out_mask_list)
        else:
            outvar = torch.zeros((self.N_t,1,self.N,self.N))
            outmask = torch.zeros_like(outvar)
        if self.squeeze:
            invar,outvar = invar.squeeze(), outvar.squeeze()
        if self.return_meta_data:
            meta = {"patch_ID":pid, "patch_coords":coords,
                    "mid_timestep":self.mid_timestep}
            return invar, outvar, meta
        elif self.return_masks:
            return invar, outvar, inmask, outmask
        return invar, outvar

    def _init_worker_local(self):
        # Initialize per-worker resources
        fs = fsspec.filesystem("file", default_fill_cache=True, block_size=128*1024**2)
        _thread_local.fs = fs

        # Lazy load SWOT swaths or numpy mask catalog
        if self.regrid_SWOT:
            _thread_local.swot_ds = [
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")),
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr"))
            ]
        else:
            _thread_local.swot_npy = np.load(
                f"{self.data_dir}/swot_npy_mask_4km.npy", mmap_mode="r")*1

        # Lazy load cloud catalog
        _thread_local.cloud_catalog = xr.open_zarr(fs.get_mapper(f"{self.data_dir}/catalog.zarr")).compute()
        _thread_local.cloud_catalog_rho = _thread_local.cloud_catalog.where( _thread_local.cloud_catalog.rho>=self.cloud_rho, drop=True)
        _thread_local.initialized = True

    def _load_fields(self, pid, fields, tkeys, mask_keys):
        vars, masks = [], []
        for fld, tk, mask_key in zip(fields, tkeys, mask_keys):
            mapper = _thread_local.fs.get_mapper(f"{self.data_dir}/{fld}_allpatches.zarr")
            ds = xr.open_zarr(mapper, consolidated=True, chunks={})
            d = ds.loc[{"patch":int(pid)}]
            d = d.isel(time=slice(self.mid_timestep - self.N_t//2,
                                  self.mid_timestep + self.N_t//2))
            if isinstance(d, xr.Dataset):
                d = next(iter(d.data_vars.values()))
            ten = self.transforms[tk](d.values)
            #ten = torch.from_numpy(arr.values).float()
            mask = self._mask_dispatch(mask_key, pid, ten.shape)
            vars.append(ten * mask)
            masks.append(mask)
        return torch.stack(vars,1), torch.stack(masks,1)

    def _mask_dispatch(self, mask_key, pid, shape):
        if (mask_key is None) or ("None" in mask_key):
            return torch.ones(shape)
        elif "swot" in mask_key.lower():
            sampling="all"
            version="random"
            if "calval" in mask_key.lower():
                version="calval"
            if "central" in mask_key.lower():
                sampling="central"
            if "random" in mask_key.lower():
                sampling="random"
            if "nadir" in mask_key.lower():
                result = (self._get_swot_mask(pid,version,sampling)+ self._get_nadir_mask(pid)) > 0
            else:
                result = self._get_swot_mask(pid,version,sampling)
        elif "nadir" in mask_key.lower():
            result = self._get_nadir_mask(pid)
        elif "cloud_rho" in mask_key.lower():
            result = self._get_cloud_rho_mask()
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")
        if self.time_loading:
            print(f"[Timer] Mask '{mask_key}' generated in {time.perf_counter() - t0:.3f} sec")
        return result

    def _get_swot_mask(self,pid,version,sampling):
        if self.regrid_SWOT:
            sw_corner, ne_corner = [-154.5, 35.3], [-147.5, 42.3]
            lat_max, lat_min, l_step, lon_i = 9000, 2000, 4, np.random.randint(5)
            lon = np.random.uniform(sw_corner[0], ne_corner[0])
            lat = np.random.uniform(sw_corner[1], ne_corner[1])
            ds = np.random.choice(_thread_local.swot_ds)
            m0 = interp_utils.grid_everything(
                _thread_local.swot_ds[0].ssha.values, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m1 = interp_utils.grid_everything(
                _thread_local.swot_ds[1].ssha.values, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m01 = np.stack([m0,m1])
        else:
            i_rand = int(np.random.uniform(64,225-64))
            j_rand = int(np.random.uniform(128,800-64))
            m01 = _thread_local.swot_npy[:,j_rand-64:j_rand+64,i_rand-64:i_rand+64]
        if int(np.random.randint(2)) < 1:
            m01 = m01[::-1,...]
        if sampling=="central":
            mask = np.zeros([self.N_t]+list(m01.shape)[-2:])
            mask[int(self.N_t/2),:,:] = m01[0]
        elif sampling=="all":
            if self.N_t > 1:
                mask_broadcast = np.broadcast_to(m01,(self.N_t//2+self.N_t%2,2,128,128))
                mask = mask_broadcast.reshape(self.N_t+self.N_t%2,128,128)[:self.N_t]
            else:
                mask = np.random.choice([m01[0],m01[1]])
        return torch.from_numpy(mask)

    def _get_nadir_mask(self, patch_ID, version="random", sample_time="1D"):
        try:
            rand_index = np.random.randint(422)
            path = f"{self.data_dir}/copernicus_nadir_SSH_daily/{rand_index:03}.zarr"
            mapper = _thread_local.fs.get_mapper(path)
            da = xr.open_zarr(mapper, consolidated=True,chunks={})
            random_tile = da.sla_filtered
        except Exception as e:
            fallback_path = f"{self.data_dir}/copernicus_nadir_SSH_daily/002.zarr"
            mapper = _thread_local.fs.get_mapper(fallback_path)
            da = xr.open_zarr(mapper, consolidated=True,chunks={})
            print(f"BAD MAPPER {rand_index} Exception: {e}")
            random_tile = da.sla_filtered
        # Temporal downsampling + slicing
        time_len = len(random_tile.time)
        mid = np.random.randint(self.N_t // 2, time_len - self.N_t // 2)
        sliced = random_tile.isel(time=slice(mid - self.N_t//2, mid + self.N_t//2 + self.N_t%2))
        # Mask where data exists
        mask = (sliced * 0 + 1).where(sliced > 0, other=0)
        return torch.from_numpy(mask.values).float()
    

    def _get_cloud_rho_mask(self):
        cc = _thread_local.cloud_catalog_rho
        masks = []
        for _ in range(self.N_t):
            samp = cc.isel(i_time=np.random.randint(len(cc.i_time)))
            pid2 = str(int(samp.patch_id.values)).zfill(3)
            mapper = _thread_local.fs.get_mapper(f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks_zarr/{pid2}.zarr")
            d = xr.open_zarr(mapper, consolidated=True)
            masks.append(~np.isnan(d.sst_filtered_q5.isel(time=int(samp.patch_timestep))).values)
        return torch.stack([torch.from_numpy(m.astype(np.float32)) for m in masks])

    def _make_standardize(self, mean=None, std=1.0):
        def fn(x):
            arr = x.values if isinstance(x, xr.DataArray) else x
            return (arr - mean)/std if mean is not None else arr/std
        return lambda da: torch.from_numpy(np.array(fn(da))).float()

    def _make_std_samplewise(self, std=1.0):
        def fn(x):
            arr = x.values if isinstance(x, xr.DataArray) else x
            return (arr - arr.mean())/std
        return lambda da: torch.from_numpy(np.array(fn(da))).float()
        
"""
class llc4320_dataset(Dataset):
    def __init__(self, data_dir, mid_timestep, N_t, patch_coords,
                 infields, outfields,
                 in_mask_list, out_mask_list,
                 in_transform_list, out_transform_list,
                 standards=None, N=128, L_x=512e3, L_y=512e3,
                 squeeze=False, return_metadata=False,
                 return_masks=False, time_loading=False,
                 regrid_SWOT=False, cloud_rho=0.7):

        self.data_dir = data_dir
        self.mid_timestep = mid_timestep
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.infields = infields
        self.outfields = outfields
        self.in_mask_list = in_mask_list
        self.out_mask_list = out_mask_list
        self.in_transform_list = in_transform_list
        self.out_transform_list = out_transform_list
        self.N = N; self.L_x = L_x; self.L_y = L_y
        self.squeeze = squeeze
        self.return_meta_data = return_metadata
        self.return_masks = return_masks
        self.time_loading = time_loading
        self.regrid_SWOT = regrid_SWOT
        self.cloud_rho = cloud_rho

        if standards is None:
            standards = {"mean_ssh":0., "std_ssh":1., "mean_sst":0., "std_sst":1.}
        self.transforms = {
            "std_ssh_norm":   self._make_standardize(std=standards["std_ssh"]),
            "std_sst_norm":   self._make_standardize(std=standards["std_sst"]),
            "std_mean_ssh_norm": self._make_std_samplewise(std=standards["std_ssh"]),
            "std_mean_sst_norm": self._make_std_samplewise(std=standards["std_sst"]),
            "std_global_mean_ssh_norm": self._make_standardize(mean=standards["mean_ssh"], std=standards["std_ssh"]),
            "std_global_mean_sst_norm": self._make_standardize(mean=standards["mean_sst"], std=standards["std_sst"]),
            "no_transform": lambda x: x
        }

    def __len__(self):
        return self.patch_coords.shape[0]

    def __getitem__(self, idx):
        pid = str(int(self.patch_coords[idx,2])).zfill(3)
        coords = self.patch_coords[idx]
        if getattr(_thread_local, 'initialized', False) is False:
            self._init_worker_local()
        invar, inmask = self._load_fields(pid,self.infields,self.in_transform_list,self.in_mask_list)
        if self.outfields:
            outvar, outmask = self._load_fields(pid, self.outfields,self.out_transform_list, self.out_mask_list)
        else:
            outvar = torch.zeros((self.N_t,1,self.N,self.N))
            outmask = torch.zeros_like(outvar)
        if self.squeeze:
            invar,outvar = invar.squeeze(), outvar.squeeze()
        if self.return_meta_data:
            meta = {"patch_ID":pid, "patch_coords":coords,
                    "mid_timestep":self.mid_timestep}
            return invar, outvar, meta
        elif self.return_masks:
            return invar, outvar, inmask, outmask
        return invar, outvar

    def _init_worker_local(self):
        # Initialize per-worker resources
        fs = fsspec.filesystem("file", default_fill_cache=True, block_size=128*1024**2)
        _thread_local.fs = fs
        # Lazy load SWOT swaths or numpy mask catalog
        if self.regrid_SWOT:
            _thread_local.swot_ds = [
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")),
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr"))
            ]
        else:
            _thread_local.swot_npy = np.load(
                f"{self.data_dir}/swot_npy_mask_4km.npy", mmap_mode="r")*1
        # Lazy load cloud catalog
        _thread_local.cloud_catalog = xr.open_zarr(fs.get_mapper(f"{self.data_dir}/catalog.zarr")).compute()
        _thread_local.cloud_catalog_rho = _thread_local.cloud_catalog.where( _thread_local.cloud_catalog.rho>=self.cloud_rho, drop=True)
        _thread_local.initialized = True

    def _load_fields(self, pid, fields, tkeys, mask_keys):
        vars, masks = [], []
        for fld, tk, mask_key in zip(fields, tkeys, mask_keys):
            mapper = _thread_local.fs.get_mapper(f"{self.data_dir}/{fld}_allpatches.zarr")
            ds = xr.open_zarr(mapper, consolidated=True)
            d = ds.loc[{"patch":int(pid)}]
            d = d.isel(time=slice(self.mid_timestep - self.N_t//2,
                                  self.mid_timestep + self.N_t//2))
            if isinstance(d, xr.Dataset):
                d = next(iter(d.data_vars.values()))
            ten = self.transforms[tk](d.values)
            #ten = torch.from_numpy(arr.values).float()
            mask = self._mask_dispatch(mask_key, pid, ten.shape)
            vars.append(ten * mask)
            masks.append(mask)
        return torch.stack(vars,1), torch.stack(masks,1)

    def _mask_dispatch(self, mask_key, pid, shape):
        if (mask_key is None) or ("None" in mask_key):
            return torch.ones(shape)
        elif "swot" in mask_key.lower():
            sampling="all"
            version="random"
            if "calval" in mask_key.lower():
                version="calval"
            if "central" in mask_key.lower():
                sampling="central"
            if "random" in mask_key.lower():
                sampling="random"
            if "nadir" in mask_key.lower():
                result = (self._get_swot_mask(pid,version,sampling)+ self._get_nadir_mask(pid)) > 0
            else:
                result = self._get_swot_mask(pid,version,sampling)
        elif "nadir" in mask_key.lower():
            result = self._get_nadir_mask(pid)
        elif "cloud_rho" in mask_key.lower():
            result = self._get_cloud_rho_mask()
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")
        if self.time_loading:
            print(f"[Timer] Mask '{mask_key}' generated in {time.perf_counter() - t0:.3f} sec")
        return result

    def _get_swot_mask(self,pid,version,sampling):
        if self.regrid_SWOT:
            sw_corner, ne_corner = [-154.5, 35.3], [-147.5, 42.3]
            lat_max, lat_min, l_step, lon_i = 9000, 2000, 4, np.random.randint(5)
            lon = np.random.uniform(sw_corner[0], ne_corner[0])
            lat = np.random.uniform(sw_corner[1], ne_corner[1])
            ds = np.random.choice(_thread_local.swot_ds)
            m0 = interp_utils.grid_everything(
                _thread_local.swot_ds[0].ssha.values, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m1 = interp_utils.grid_everything(
                _thread_local.swot_ds[1].ssha.values, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m01 = np.stack([m0,m1])
        else:
            i_rand = int(np.random.uniform(64,225-64))
            j_rand = int(np.random.uniform(128,800-64))
            m01 = _thread_local.swot_npy[:,j_rand-64:j_rand+64,i_rand-64:i_rand+64]
        if int(np.random.randint(2)) < 1:
            m01 = m01[::-1,...]
        if sampling=="central":
            mask = np.zeros([self.N_t]+list(m01.shape)[-2:])
            mask[int(self.N_t/2),:,:] = m01[0]
        elif sampling=="all":
            if self.N_t > 1:
                mask_broadcast = np.broadcast_to(m01,(self.N_t//2+self.N_t%2,2,128,128))
                mask = mask_broadcast.reshape(self.N_t+self.N_t%2,128,128)[:self.N_t]
            else:
                mask = np.random.choice([m01[0],m01[1]])
        return torch.from_numpy(mask)

    def _get_nadir_mask(self, patch_ID, version="random", sample_time="1D"):
        try:
            rand_index = np.random.randint(422)
            path = f"{self.data_dir}/copernicus_nadir_SSH_daily/{rand_index:03}.zarr"
            mapper = _thread_local.fs.get_mapper(path)
            da = xr.open_zarr(mapper, consolidated=True)
            random_tile = da.sla_filtered
        except Exception as e:
            fallback_path = f"{self.data_dir}/copernicus_nadir_SSH_daily/002.zarr"
            mapper = _thread_local.fs.get_mapper(fallback_path)
            da = xr.open_zarr(mapper, consolidated=True)
            print(f"BAD MAPPER {rand_index} Exception: {e}")
            random_tile = da.sla_filtered
        # Temporal downsampling + slicing
        time_len = len(random_tile.time)
        mid = np.random.randint(self.N_t // 2, time_len - self.N_t // 2)
        sliced = random_tile.isel(time=slice(mid - self.N_t//2, mid + self.N_t//2 + self.N_t%2))
        # Mask where data exists
        mask = (sliced * 0 + 1).where(sliced > 0, other=0)
        return torch.from_numpy(mask.values).float()
    

    def _get_cloud_rho_mask(self):
        cc = _thread_local.cloud_catalog_rho
        masks = []
        for _ in range(self.N_t):
            samp = cc.isel(i_time=np.random.randint(len(cc.i_time)))
            pid2 = str(int(samp.patch_id.values)).zfill(3)
            mapper = _thread_local.fs.get_mapper(f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks_zarr/{pid2}.zarr")
            d = xr.open_zarr(mapper, consolidated=True)
            masks.append(~np.isnan(d.sst_filtered_q5.isel(time=int(samp.patch_timestep))).values)
        return torch.stack([torch.from_numpy(m.astype(np.float32)) for m in masks])

    def _make_standardize(self, mean=None, std=1.0):
        def fn(x):
            arr = x.values if isinstance(x, xr.DataArray) else x
            return (arr - mean)/std if mean is not None else arr/std
        return lambda da: torch.from_numpy(np.array(fn(da))).float()

    def _make_std_samplewise(self, std=1.0):
        def fn(x):
            arr = x.values if isinstance(x, xr.DataArray) else x
            return (arr - arr.mean())/std
        return lambda da: torch.from_numpy(np.array(fn(da))).float()

"""