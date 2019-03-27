package com.nigam.ancache.core;

import java.util.NoSuchElementException;
import java.util.Set;

import com.nigam.ancache.model.CacheElement;

public interface ANCache<K, V> {
	
	CacheElement<K, V> add(K k, V v);
	
	CacheElement<K, V> update(V v) throws NoSuchElementException;
	
	CacheElement<K, V> addOrUpdate(K k, V v);
	
	Boolean evictByKey(K k);
	
	Boolean evictByValue(V v) throws NoSuchElementException;
	
	Set<CacheElement<K,V>> searchByValue(V v);
	
	CacheElement<K,V> get(K k);

}
